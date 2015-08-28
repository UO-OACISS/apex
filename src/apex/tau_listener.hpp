//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TAUHANDLER_HPP
#define TAUHANDLER_HPP

#include "event_listener.hpp"
#include <memory>

namespace apex {

class tau_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
public:
  tau_listener (void);
  ~tau_listener (void) { };
  void on_startup(startup_event_data &data);
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_exit_thread(event_data &data);
  bool on_start(apex_function_address function_address);
  bool on_start(std::string *timer_name);
  void on_stop(std::shared_ptr<profiler> p);
  void on_yield(std::shared_ptr<profiler> p);
  bool on_resume(apex_function_address function_address);
  bool on_resume(std::string *timer_name);
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &data);

};

}

#endif // TAUHANDLER_HPP
