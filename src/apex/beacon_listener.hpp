//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BEACON_HANDLER_HPP
#define BEACON_HANDLER_HPP

#include "event_listener.hpp"
#include "beacon_event.hpp"
#include <boost/array.hpp>
#include <boost/asio.hpp>

namespace apex {

class beacon_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
  boost::asio::io_service _io_service;
  boost::asio::ip::udp::endpoint _receiver_endpoint;
  boost::asio::ip::udp::socket * _socket;
  void synchronize_message(apex_beacon_event event);
 public:
  beacon_listener (void);
  ~beacon_listener (void) { };
  void on_startup(startup_event_data &data);
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_start(timer_event_data &data);
  void on_stop(timer_event_data &data);
  void on_yield(timer_event_data &data);
  void on_resume(timer_event_data &data);
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &event_data);

};

}

#endif // BEACON_HANDLER_HPP
