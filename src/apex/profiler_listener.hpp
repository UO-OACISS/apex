//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROFILER_LISTENER_HPP
#define PROFILER_LISTENER_HPP

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "event_listener.hpp"
#include "apex_types.h"
#include <boost/atomic.hpp>
#include <vector>
#include <boost/thread.hpp>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "profile.hpp"
#include "udp_client.hpp"

namespace apex {

class profiler_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
  static boost::atomic<int> active_tasks;
  static profiler * main_timer;
  static void finalize_profiles(void);
  static void write_profile(int tid);
  static void delete_profiles(void);
  static void process_profiles(profiler_listener * listener);
  static unsigned int process_profile(profiler * p, unsigned int tid);
  static int node_id;
  static boost::mutex _mtx;
  void _common_start(timer_event_data &data, bool is_resume); // internal, inline function
  void _common_stop(timer_event_data &data, bool is_yield); // internal, inline function
  udp_client client;
public:
  profiler_listener (void)  : _terminate(false) {
      if (apex_options::use_beacon()) {
        client.start_client();
      }
  };
  ~profiler_listener (void) { 
      if (apex_options::use_beacon()) {
        client.stop_client();
      }
  };
  // events
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
  // other methods
  static void reset(apex_function_address function_address);
  static void reset(const std::string &timer_name);
  static profile * get_profile(apex_function_address address);
  static profile * get_profile(const std::string &timer_name);
  static std::vector<std::string> get_available_profiles();
};

}

#endif // PROFILER_LISTENER_HPP
