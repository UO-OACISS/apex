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

void tau_listener::on_startup(startup_event_data &data) {
  if (!_terminate) {
      TAU_PROFILE_INIT(data.argc, data.argv);
  }
  return;
}

void tau_listener::on_shutdown(shutdown_event_data &data) {
  if (!_terminate) {
      _terminate = true;
      Tau_profile_exit_all_threads();
      TAU_PROFILE_EXIT("APEX exiting");
  }
  return;
}

void tau_listener::on_new_node(node_event_data &data) {
  if (!_terminate) {
      TAU_PROFILE_SET_NODE(data.node_id);
  }
  return;
}

void tau_listener::on_new_thread(new_thread_event_data &data) {
  if (!_terminate) {
      TAU_REGISTER_THREAD();
      Tau_create_top_level_timer_if_necessary();
      // set the thread id for future listeners to this event
      data.thread_id = TAU_PROFILE_GET_THREAD();
  }
  return;
}

void tau_listener::on_start(apex_function_address function_address, string *timer_name) {
  if (!_terminate) {
      if (timer_name != NULL) {
      	TAU_START(timer_name->c_str());
      } else {
      	TAU_START(thread_instance::instance().map_addr_to_name(function_address).c_str());
      }
  }
  return;
}

void tau_listener::on_stop(profiler *p) {
  static string empty("");
  if (!_terminate) {
      if (p->have_name) {
        if (p->timer_name->compare(empty) == 0) {
          TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
	} else {
          TAU_STOP(p->timer_name->c_str());
	}
      } else {
        if (p->action_address == 0) {
          TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
	} else {
      	  TAU_STOP(thread_instance::instance().map_addr_to_name(p->action_address).c_str());
	}
      }
  }
  return;
}

void tau_listener::on_resume(profiler *p) {
  if (!_terminate) {
    if (p->have_name) {
      TAU_START(p->timer_name->c_str());
    } else {
      TAU_START(thread_instance::instance().map_addr_to_name(p->action_address).c_str());
    }
  }
  return;
}

void tau_listener::on_sample_value(sample_value_event_data &data) {
  if (!_terminate) {
      Tau_trigger_context_event_thread((char*)data.counter_name->c_str(), data.counter_value, data.thread_id);
  }
  return;
}

void tau_listener::on_periodic(periodic_event_data &data) {
  if (!_terminate) {
  }
  return;
}

void tau_listener::on_custom_event(custom_event_data &data) {
  if (!_terminate) {
  }
  return;
}


}
