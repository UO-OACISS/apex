//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UDP_CLIENT_HPP
#define UDP_CLIENT_HPP

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "apex_types.h"
#include "profile.hpp"
#include <boost/asio.hpp>
#include <map>

namespace apex {

class udp_client {
private:
  static boost::asio::io_service _io_service;
  static boost::asio::ip::udp::endpoint _receiver_endpoint;
  static boost::asio::ip::udp::socket * _socket;
public:
  static void synchronize_profiles(std::map<std::string, profile*> name_map, std::map<apex_function_address, profile*> address_map);
  static void start_client(void);
  static void stop_client(void);
  udp_client (void) { };
  ~udp_client (void) { };
};

}

#endif // UDP_CLIENT_HPP
