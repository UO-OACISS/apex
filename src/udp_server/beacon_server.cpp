//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include "apex_types.h"
#include "apex_options.hpp"
#include "beacon_event.hpp"
#include "profiler_listener.hpp"

bool shutdown_flag = false;

using boost::asio::ip::udp;

class udp_server
{
public:
  udp_server(boost::asio::io_service& io_service)
    : socket_(io_service, udp::endpoint(udp::v4(), atoi(apex::apex_options::beacon_port())))
  {
    start_receive();
    listener = new apex::profiler_listener();
    apex::startup_event_data data(0, NULL);
    listener->on_startup(data);
  }

private:
  void start_receive()
  {
    socket_.async_receive_from(
        boost::asio::buffer(recv_buffer_, 1024), remote_endpoint_,
        boost::bind(&udp_server::handle_receive, this,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
  }

  void handle_receive(const boost::system::error_code& error,
      std::size_t bytes_transferred)
  {
    if (!error || error == boost::asio::error::message_size)
    {
        try {
            std::string archive_data(&recv_buffer_[0], bytes_transferred);
            std::istringstream archive_stream(archive_data);
            boost::archive::text_iarchive archive(archive_stream);
            apex::apex_beacon_event e;
            archive >> e;
            switch(e.event_type) {
              case APEX_STARTUP:
              {
                std::cout << "Got startup from client. " << std::endl;
                apex::startup_event_data data(0, NULL);
                //listener->on_startup(data);
                break;
              }
              case APEX_SHUTDOWN:
              {
                std::cout << "Got shutdown from client. " << std::endl;
                apex::shutdown_event_data data(0,0);
                listener->on_shutdown(data);
                if (shutdown_flag) {
                    std::cout << "Exiting as requested" << std::endl;
                    exit (APEX_NOERROR);
                }
                break;
              }
              case APEX_NEW_NODE:
              {
                std::cout << "Got new node from client. " << std::endl;
                //apex::node_event_data data(0,0);
                //listener->on_new_node(data);
                break;
              }
              case APEX_NEW_THREAD:
              {
                std::cout << "Got new thread from client. " << std::endl;
                //apex::new_thread_event_data data("");
                //listener->on_new_thread(data);
                break;
              }
              case APEX_STOP_EVENT:
              {
                //std::cout << "Got stop event from client. " << e.value << std::endl;
                apex::sample_value_event_data data(0, e.name, e.value);
                data.is_counter = false;
                listener->on_sample_value(data);
                break;
              }
              case APEX_SAMPLE_VALUE:
              {
                //std::cout << "Got sampled value from client. " << std::endl;
                apex::sample_value_event_data data(0, e.name, e.value);
                listener->on_sample_value(data);
                break;
              }
              default:
              {
                break;
              }
            }
         } catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
        start_receive();
    }
  }

  void handle_send(boost::shared_ptr<std::string> /*message*/,
      const boost::system::error_code& /*error*/,
      std::size_t /*bytes_transferred*/)
  {
  }

  udp::socket socket_;
  udp::endpoint remote_endpoint_;
  char recv_buffer_[1024];
  apex::profiler_listener * listener;
};

int main(int argc, char** argv) {
  try
  {
      namespace po = boost::program_options; 
      po::options_description desc("Options"); 
      desc.add_options() 
      ("help,h", "Print help messages") 
      ("shutdown,s", "Exit when shutdown event received");

      po::variables_map vm;
      po::store(po::parse_command_line(argc,argv,desc), vm);
      po::notify(vm);

      if(vm.count("help")) {
          std::cout << desc << std::endl;
          return APEX_NOERROR;
      }

      if(vm.count("shutdown")) {
          shutdown_flag = true;
      }
               
    boost::asio::io_service io_service;
    udp_server server(io_service);
    io_service.run();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
