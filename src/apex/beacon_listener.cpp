//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "beacon_listener.hpp"
#include <boost/archive/text_oarchive.hpp>
#include "thread_instance.hpp"
#include "apex_options.hpp"
#include "address_resolution.hpp"
#include <iostream>

using namespace std;
using boost::asio::ip::udp;

namespace apex {

beacon_listener::beacon_listener (void) : _terminate(false) {
    try {
        boost::asio::ip::udp::resolver _resolver(_io_service);
        boost::asio::ip::udp::resolver::query _query(udp::v4(), apex_options::beacon_host(), apex_options::beacon_port());
        _receiver_endpoint = *_resolver.resolve(_query);
        _socket = new boost::asio::ip::udp::socket(_io_service);
        _socket->open(udp::v4());
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

inline void beacon_listener::synchronize_message(apex_beacon_event event) {
    std::stringstream ss;
    boost::archive::text_oarchive archive(ss);
    archive << event;
    //std::string outbound_data_ = ss.str();
    _socket->send_to(boost::asio::buffer(ss.str()), _receiver_endpoint);
}

void beacon_listener::on_startup(startup_event_data &data) {
    APEX_UNUSED(data);
    apex_beacon_event e;
    e.event_type = APEX_STARTUP;
    synchronize_message(e);
  return;
}

void beacon_listener::on_shutdown(shutdown_event_data &data) {
  APEX_UNUSED(data);
  if (_terminate) {
    return;
  }
  _terminate = true;
  apex_beacon_event e;
  e.event_type = APEX_SHUTDOWN;
  synchronize_message(e);
  return;
}

void beacon_listener::on_new_node(node_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
    apex_beacon_event e;
    e.event_type = APEX_NEW_NODE;
    synchronize_message(e);
  }
  return;
}

void beacon_listener::on_new_thread(new_thread_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
    apex_beacon_event e;
    e.event_type = APEX_NEW_THREAD;
    synchronize_message(e);
  }
  return;
}

void beacon_listener::on_start(timer_event_data &data) {
  APEX_UNUSED(data);
  return;
}

void beacon_listener::on_stop(timer_event_data &data) {
  if (!_terminate) {
    apex_beacon_event e;
    e.event_type = APEX_STOP_EVENT;
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(data.end_timestamp - data.start_timestamp);
    e.value = time_span.count();
    e.have_name = data.have_name;
    if (data.have_name) {
        e.name = string(data.timer_name->c_str());
    } else {
        e.function_address = data.function_address;
        //e.name = thread_instance::instance().map_addr_to_name(data.function_address);
        e.name = *(lookup_address((uintptr_t)data.function_address, false));
    }
    //std::cout << e.name << " timer: " << e.value << std::endl;
    synchronize_message(e);
  }
  return;
}

void beacon_listener::on_yield(timer_event_data &data) {
  APEX_UNUSED(data);
  return;
}

void beacon_listener::on_resume(timer_event_data &data) {
  APEX_UNUSED(data);
  return;
}

void beacon_listener::on_sample_value(sample_value_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
    apex_beacon_event e;
    e.event_type = APEX_SAMPLE_VALUE;
    e.have_name = true;
    e.name = string(data.counter_name->c_str());
    e.value = data.counter_value;
    synchronize_message(e);
  }
  return;
}

void beacon_listener::on_periodic(periodic_event_data &data) {
  APEX_UNUSED(data);
  return;
}

void beacon_listener::on_custom_event(custom_event_data &data) {
  APEX_UNUSED(data);
  return;
}


}
