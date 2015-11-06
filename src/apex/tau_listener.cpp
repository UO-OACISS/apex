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

void tau_listener::initialize_tau(int argc, char** argv) {
  if (argc == 0) {
    int _argc = 1;
    const char *_dummy = "APEX Application";
    char* _argv[1];
    _argv[0] = const_cast<char*>(_dummy);
    TAU_PROFILE_INIT(_argc, _argv);
  } else {
    TAU_PROFILE_INIT(argc, argv);
  }
  TAU_PROFILE_SET_NODE(0);
  Tau_create_top_level_timer_if_necessary();
}

tau_listener::tau_listener (void) : _terminate(false) {
}

void tau_listener::on_startup(startup_event_data &data) {
/*
  if (!_terminate) {
      TAU_PROFILE_INIT(data.argc, data.argv);
      TAU_PROFILE_SET_NODE(0);
      Tau_create_top_level_timer_if_necessary();
  }
  */
  return;
}

void tau_listener::on_shutdown(shutdown_event_data &data) {
  APEX_UNUSED(data);
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

void tau_listener::on_exit_thread(event_data &data) {
  if (!_terminate) {
      //TAU_PROFILE_EXIT("APEX exiting");
    //std::cout << "TAU_EXIT_THREAD" << std::endl;
  }
  APEX_UNUSED(data);
  return;
}

bool tau_listener::on_start(apex_function_address function_address) {
  if (!_terminate) {
    TAU_START(thread_instance::instance().map_addr_to_name(function_address).c_str());
  } else {
      return false;
  }
  return true;
}

bool tau_listener::on_start(std::string * timer_name) {
  if (!_terminate) {
    TAU_START(timer_name->c_str());
  } else {
      return false;
  }
  return true;
}

bool tau_listener::on_resume(apex_function_address function_address) {
  if (!_terminate) {
    TAU_START(thread_instance::instance().map_addr_to_name(function_address).c_str());
    return true;
  }
  return false;
}

bool tau_listener::on_resume(std::string * timer_name) {
  if (!_terminate) {
    TAU_START(timer_name->c_str());
    return true;
  }
  return false;
}

void tau_listener::on_stop(std::shared_ptr<profiler> p) {
  static string empty("");
  if (!_terminate) {
      if (p->have_name) {
        if (p->timer_name->compare(empty) == 0) {
          //printf("TAU stopping: GLOBAL\n");
          TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
        } else {
          //printf("TAU stopping: '%s'\n", p->timer_name->c_str());
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

void tau_listener::on_yield(std::shared_ptr<profiler> p) {
    on_stop(p);
}

void tau_listener::on_sample_value(sample_value_event_data &data) {
  if (!_terminate) {
      Tau_trigger_context_event_thread(const_cast<char*>(data.counter_name->c_str()), data.counter_value, data.thread_id);
  }
  return;
}

void tau_listener::on_periodic(periodic_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
  }
  return;
}

void tau_listener::on_custom_event(custom_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
  }
  return;
}


}
