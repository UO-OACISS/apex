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
#include "beacon_event.hpp"

using boost::asio::ip::udp;

class udp_server
{
public:
  udp_server(boost::asio::io_service& io_service)
    : socket_(io_service, udp::endpoint(udp::v4(), 5560))
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
            if (e.have_name && e.event_type == APEX_SAMPLE_VALUE) {
                std::cout << "Got message from client: " << e.name << std::endl;
            } else {
                std::cout << "Got message from client. " << std::endl;
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
