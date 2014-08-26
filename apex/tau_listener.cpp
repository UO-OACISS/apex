//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tau_listener.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <fstream>

#define PROFILING_ON
//#define TAU_GNU
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>

extern "C" int Tau_profile_exit_all_tasks();
extern "C" int Tau_profile_exit_all_threads();

using namespace std;

namespace apex {

tau_listener::tau_listener (void) : _terminate(false) {
}

void tau_listener::on_event(event_data * event_data_) {
  unsigned int tid = thread_instance::get_id();
  if (!_terminate) {
    if (event_data_->event_type_ == START_EVENT) {
      timer_event_data *tmp = (timer_event_data*)event_data_;
      TAU_START(tmp->timer_name->c_str());
    } else if (event_data_->event_type_ == STOP_EVENT) {
      timer_event_data *tmp = (timer_event_data*)event_data_;
      if (*(tmp->timer_name) == string("")) {
        TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
      } else {
        TAU_STOP(tmp->timer_name->c_str());
      }
    } else if (event_data_->event_type_ == NEW_NODE) {
      node_event_data *tmp = (node_event_data*)event_data_;
      TAU_PROFILE_SET_NODE(tmp->node_id);
    } else if (event_data_->event_type_ == NEW_THREAD) {
      TAU_REGISTER_THREAD();
      // set the thread id for future listeners to this event
      event_data_->thread_id = TAU_PROFILE_GET_THREAD();
    } else if (event_data_->event_type_ == SAMPLE_VALUE) {
      sample_value_event_data *tmp = (sample_value_event_data*)event_data_;
      Tau_trigger_context_event_thread((char*)tmp->counter_name->c_str(), tmp->counter_value, tmp->thread_id);
    } else if (event_data_->event_type_ == STARTUP) {
      startup_event_data *tmp = (startup_event_data*)event_data_;
      TAU_PROFILE_INIT(tmp->argc, tmp->argv);
    } else if (event_data_->event_type_ == SHUTDOWN) {
      shutdown_event_data *tmp = (shutdown_event_data*)event_data_;
      _terminate = true;
      Tau_profile_exit_all_threads();
      TAU_PROFILE_EXIT("APEX exiting");
    }
  }
  return;
}

}
