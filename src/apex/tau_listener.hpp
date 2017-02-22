//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

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
  static void initialize_tau(int argc, char** avgv);
  void on_startup(startup_event_data &data);
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_exit_thread(event_data &data);
  bool on_start(task_identifier *id);
  void on_stop(std::shared_ptr<profiler> &p);
  void on_yield(std::shared_ptr<profiler> &p);
  bool on_resume(task_identifier * id);
  void on_new_task(task_identifier * id, uint64_t task_id)
       { APEX_UNUSED(id); APEX_UNUSED(task_id); };
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &data);
  void on_send(message_event_data &data) { APEX_UNUSED(data); };
  void on_recv(message_event_data &data) { APEX_UNUSED(data); };

};

}

/* Weak symbols that are redefined if we load TAU at link or runtime */
extern "C" {
int __attribute__((weak)) initialize_worker_thread_for_tau(void);
int __attribute__((weak)) Tau_register_thread(void);
int __attribute__((weak)) Tau_create_top_level_timer_if_necessary(void);
int __attribute__((weak)) Tau_start(const char *);
int __attribute__((weak)) Tau_stop(const char *);
int __attribute__((weak)) Tau_init(int, char**);
int __attribute__((weak)) Tau_exit(const char*);
int __attribute__((weak)) Tau_set_node(int);
int __attribute__((weak)) Tau_profile_exit_all_threads(void);
int __attribute__((weak)) Tau_get_thread(void);
int __attribute__((weak)) Tau_profile_exit_all_tasks(void);
int __attribute__((weak)) Tau_global_stop(void);
int __attribute__((weak)) Tau_trigger_context_event_thread(char*, double, int);
}
