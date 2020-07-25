//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "event_listener.hpp"
#include <memory>

namespace apex {

class trace_event_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
  bool _common_start(std::shared_ptr<task_wrapper> &tt_ptr);
  void _common_stop(std::shared_ptr<profiler> &p);
  static bool _initialized;
public:
  trace_event_listener (void);
  ~trace_event_listener (void) { };
  static bool initialize_tau(int argc, char** avgv);
  inline static bool initialized(void) { return _initialized; }
  void on_startup(startup_event_data &data);
  void on_dump(dump_event_data &data);
  void on_reset(task_identifier * id)
      { APEX_UNUSED(id); };
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_exit_thread(event_data &data);
  bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
  void on_stop(std::shared_ptr<profiler> &p);
  void on_yield(std::shared_ptr<profiler> &p);
  bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
  void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
  };
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &data);
  void on_send(message_event_data &data) { APEX_UNUSED(data); };
  void on_recv(message_event_data &data) { APEX_UNUSED(data); };
  void set_node_id(int node_id, int node_count);
  void set_metadata(const char * name, const char * value);

  static void Tau_start_wrapper(const char * name);
  static void Tau_stop_wrapper(const char * name);
};

int initialize_worker_thread_for_tau(void);

}

