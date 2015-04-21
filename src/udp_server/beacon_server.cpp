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
#include "apex_types.h"
#include "apex_options.hpp"
#include "beacon_event.hpp"

using boost::asio::ip::udp;

class udp_server
{
public:
  udp_server(boost::asio::io_service& io_service)
    : socket_(io_service, udp::endpoint(udp::v4(), atoi(apex::apex_options::beacon_port())))
  {
    start_receive();
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
                std::cout << "Got startup from client. " << std::endl;
                break;
              case APEX_SHUTDOWN:
                std::cout << "Got shutdown from client. " << std::endl;
                break;
              case APEX_NEW_NODE:
                std::cout << "Got new node from client. " << std::endl;
                break;
              case APEX_NEW_THREAD:
                std::cout << "Got new thread from client. " << std::endl;
                break;
              case APEX_START_EVENT:
                std::cout << "Got start event from client. " << std::endl;
                break;
              case APEX_RESUME_EVENT:
                std::cout << "Got resume event from client. " << std::endl;
                break;
              case APEX_STOP_EVENT:
                std::cout << "Got stop event from client. " << std::endl;
                break;
              case APEX_YIELD_EVENT:
                std::cout << "Got yield event from client. " << std::endl;
                break;
              case APEX_SAMPLE_VALUE:
                std::cout << "Got sample value from client. " << std::endl;
                break;
              case APEX_PERIODIC:
                std::cout << "Got periodic from client. " << std::endl;
                break;
              case APEX_CUSTOM_EVENT:
                std::cout << "Got custom event from client. " << std::endl;
                break;
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
};

int main()
{
  try
  {
    boost::asio::io_service io_service;
    udp_server server(io_service);
    io_service.run();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
