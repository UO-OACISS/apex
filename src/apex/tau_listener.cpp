//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tau_listener.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <fstream>
#include <memory>

using namespace std;

namespace apex {

int (*my_Tau_init)(int, char**);
int (*my_Tau_register_thread)(void);
int (*my_Tau_create_top_level_timer_if_necessary)(void);
int (*my_Tau_start)(const char *);
int (*my_Tau_stop)(const char *);
int (*my_Tau_exit)(const char*);
int (*my_Tau_set_node)(int);
int (*my_Tau_dump)(void);
int (*my_Tau_profile_exit_all_threads)(void);
int (*my_Tau_get_thread)(void);
int (*my_Tau_profile_exit_all_tasks)(void);
int (*my_Tau_global_stop)(void);
int (*my_Tau_trigger_context_event_thread)(char*, double, int);

bool assign_function_pointers(void) {
  if (Tau_init == nullptr) {
    /* Print an error message, because TAU wasn't preloaded! */
    std::cerr <<
        "WARNING! TAU libraries not loaded, TAU support unavailable!"
        << std::endl;
    return false;
  } else {
    my_Tau_init = &Tau_init;
  }
  if (Tau_register_thread != nullptr) {
    my_Tau_register_thread = &Tau_register_thread;
  }
  if (Tau_create_top_level_timer_if_necessary != nullptr) {
    my_Tau_create_top_level_timer_if_necessary = &Tau_create_top_level_timer_if_necessary;
  }
  if (Tau_start != nullptr) {
    my_Tau_start = &Tau_start;
  }
  if (Tau_stop != nullptr) {
    my_Tau_stop = &Tau_stop;
  }
  if (Tau_dump != nullptr) {
    my_Tau_dump = &Tau_dump;
  }
  if (Tau_exit != nullptr) {
    my_Tau_exit = &Tau_exit;
  }
  if (Tau_set_node != nullptr) {
    my_Tau_set_node = &Tau_set_node;
  }
  if (Tau_profile_exit_all_threads != nullptr) {
    my_Tau_profile_exit_all_threads = &Tau_profile_exit_all_threads;
  }
  if (Tau_get_thread != nullptr) {
    my_Tau_get_thread = &Tau_get_thread;
  }
  if (Tau_profile_exit_all_tasks != nullptr) {
    my_Tau_profile_exit_all_tasks = &Tau_profile_exit_all_tasks;
  }
  if (Tau_global_stop != nullptr) {
    my_Tau_global_stop = &Tau_global_stop;
  }
  if (Tau_trigger_context_event_thread != nullptr) {
    my_Tau_trigger_context_event_thread = &Tau_trigger_context_event_thread;
  }
  return true;
}

bool tau_listener::initialize_tau(int argc, char** argv) {
  if (assign_function_pointers() == false) { return false; }
  if (argc == 0) {
    int _argc = 1;
    const char *_dummy = "APEX Application";
    char* _argv[1];
    _argv[0] = const_cast<char*>(_dummy);
    //if (apex::apex_options::use_tau()) {
    my_Tau_init(_argc, _argv);
  } else {
    my_Tau_init(argc, argv);
  }
  my_Tau_set_node(0);
  my_Tau_create_top_level_timer_if_necessary();
  return true;
}

tau_listener::tau_listener (void) : _terminate(false) {
}

void tau_listener::on_startup(startup_event_data &data) {
  return;
}

void tau_listener::on_dump(dump_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
      my_Tau_dump();
  }
  return;
}

void tau_listener::on_shutdown(shutdown_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
      _terminate = true;
      my_Tau_profile_exit_all_threads();
      my_Tau_exit("APEX exiting");
  }
  return;
}

void tau_listener::on_new_node(node_event_data &data) {
  if (!_terminate) {
      my_Tau_set_node(data.node_id);
  }
  return;
}

void tau_listener::on_new_thread(new_thread_event_data &data) {
  if (!_terminate) {
      my_Tau_register_thread();
      my_Tau_create_top_level_timer_if_necessary();
      // set the thread id for future listeners to this event
      data.thread_id = my_Tau_get_thread();
  }
  return;
}

void tau_listener::on_exit_thread(event_data &data) {
  if (!_terminate) {
  }
  APEX_UNUSED(data);
  return;
}

inline bool tau_listener::_common_start(std::shared_ptr<task_wrapper> &tt_ptr) {
  if (!_terminate) {
    const char * tmp = tt_ptr->get_task_id()->get_name().c_str();
    //printf("Starting: %s\n", tmp);
    my_Tau_start(tmp);
  } else {
      return false;
  }
  return true;
}

bool tau_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
  return _common_start(tt_ptr);
}

bool tau_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
  return _common_start(tt_ptr);
}

inline void tau_listener::_common_stop(std::shared_ptr<profiler> &p) {
  static string empty("");
  if (!_terminate) {
      //if (p->tt_ptr->get_task_id()->get_name().compare(empty) == 0) {
          my_Tau_global_stop(); // stop the top level timer
          /*
      } else {
          const char * tmp = p->tt_ptr->get_task_id()->get_name().c_str();
          printf("Stopping: %s\n", tmp);
          my_Tau_stop(tmp);
      }
      */
  }
  return;
}

void tau_listener::on_stop(std::shared_ptr<profiler> &p) {
  return _common_stop(p);
}

void tau_listener::on_yield(std::shared_ptr<profiler> &p) {
  return _common_stop(p);
}

void tau_listener::on_sample_value(sample_value_event_data &data) {
  if (!_terminate) {
      my_Tau_trigger_context_event_thread(
        const_cast<char*>(data.counter_name->c_str()),
        data.counter_value, data.thread_id);
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

void tau_listener::set_node_id(int node_id, int node_count) {
  my_Tau_set_node(node_id);
}

/* This function is used by APEX threads so that TAU knows about them. */
int initialize_worker_thread_for_tau(void) {
  if (apex_options::use_tau())
  {
    my_Tau_register_thread();
    my_Tau_create_top_level_timer_if_necessary();
  }
  return 0;
}

}

