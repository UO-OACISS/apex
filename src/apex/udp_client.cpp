//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "udp_client.hpp"
#include "apex_types.h"
#include "apex_api.hpp"
#include "thread_instance.hpp"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using namespace std;
using namespace apex;
using boost::asio::ip::udp;

namespace apex {

  void udp_client::start_client(void) {
    try {
        std::cout << apex_options::beacon_clientip() << " connecting to " << apex_options::beacon_host() << ":" << apex_options::beacon_port() << std::endl;
        boost::asio::ip::udp::resolver resolver(_io_service);
        boost::asio::ip::udp::resolver::query query_remote(udp::v4(), apex_options::beacon_host(), apex_options::beacon_port());
        _receiver_endpoint = *resolver.resolve(query_remote);
        _socket = new boost::asio::ip::udp::socket(_io_service);
        _socket->open(udp::v4());
        if (strlen(apex_options::beacon_clientip()) > 0) {
            boost::asio::ip::udp::endpoint localEndpoint(
                        boost::asio::ip::address::from_string(apex_options::beacon_clientip()), 0);
            //boost::asio::ip::udp::resolver::query query_local(udp::v4(), apex_options::beacon_clientip());
            //boost::asio::ip::udp::endpoint localEndpoint = *resolver.resolve(query_local);
            _socket->bind(localEndpoint);
        }
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
  }

  std::string profile_to_string(std::string& name, apex_profile * p) {
        std::stringstream ss;
        ss << name.length() << ":" << name << ":";
        ss << p->calls << ":";
        ss << p->accumulated << ":";
        ss << p->sum_squares << ":";
        ss << p->minimum << ":";
        ss << p->maximum << ":";
        return ss.str();
    }

    void profile_to_json(std::string& name, rapidjson::Writer<rapidjson::StringBuffer> &writer, apex_profile * p) {
        writer.StartObject();
        writer.String(name.c_str());
        writer.StartArray();
        writer.Double(p->calls);
        writer.Double(p->accumulated);
        writer.Double(p->minimum);
        writer.Double(p->maximum);
        writer.Double(p->sum_squares);
        writer.EndArray();
        writer.EndObject();
    }

    void write_json_header(rapidjson::Writer<rapidjson::StringBuffer> &writer) {
        writer.StartObject();
        writer.Key("Profile Format");
        writer.String("APEX");
        writer.Key("Version");
        writer.String(version().c_str());
    }

    void write_json_footer(rapidjson::Writer<rapidjson::StringBuffer> &writer) {
        writer.EndObject();
    }

  void udp_client::synchronize_profiles(std::map<std::string, profile*> name_map, std::map<apex_function_address, profile*> address_map) {
    //std::stringstream ss;
    rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
    write_json_header(writer);
    writer.String("timers");
    writer.StartArray();
    // iterate over the string map
    for(std::map<string, profile*>::iterator it = name_map.begin(); it != name_map.end(); it++) {
        std::string name = it->first;
        profile * p = it->second;
        if (p->get_profile()->type == APEX_TIMER) {
            profile_to_json(name, writer, p->get_profile());
        }
    }
    for(std::map<apex_function_address, profile*>::iterator it = address_map.begin(); it != address_map.end(); it++) {
        apex_function_address address = it->first;
        profile * p = it->second;
#if defined(HAVE_BFD)
        std::string * name = lookup_address((uintptr_t)address, false);
        profile_to_json(*name, writer, p->get_profile());
        delete name;
#else
        std::string name = thread_instance::instance().map_addr_to_name(address);
        profile_to_json(name, writer, p->get_profile());
#endif
    }
    writer.EndArray();
    writer.String("counters");
    writer.StartArray();
    // iterate over the string map
    for(std::map<string, profile*>::iterator it = name_map.begin(); it != name_map.end(); it++) {
        std::string name = it->first;
        profile * p = it->second;
        if (p->get_profile()->type == APEX_COUNTER) {
            profile_to_json(name, writer, p->get_profile());
        }
    }
    writer.EndArray();
    write_json_footer(writer);
    std::string outstring(s.GetString());
    //std::cout << "Sending profiles (bytes): " << outstring.length() << std::endl;
    if (outstring.length() > 64*1024) { // max UDP length
        std::cerr << "Warning! Profile data exceeds allowable transmission size!" << std::endl;
    }
    _socket->send_to(boost::asio::buffer(outstring), _receiver_endpoint);
  }

  void udp_client::stop_client() {
    std::string outstring("");
    _socket->send_to(boost::asio::buffer(outstring), _receiver_endpoint);
  }

}
