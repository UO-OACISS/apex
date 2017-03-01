//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tau_listener.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <fstream>

using namespace std;

namespace apex {

void tau_listener::initialize_tau(int argc, char** argv) {
  if (argc == 0) {
    int _argc = 1;
    const char *_dummy = "APEX Application";
    char* _argv[1];
    _argv[0] = const_cast<char*>(_dummy);
    Tau_init(_argc, _argv);
  } else {
    Tau_init(argc, argv);
  }
  Tau_set_node(0);
  Tau_create_top_level_timer_if_necessary();
}

tau_listener::tau_listener (void) : _terminate(false) {
}

void tau_listener::on_startup(startup_event_data &data) {
  return;
}

void tau_listener::on_shutdown(shutdown_event_data &data) {
  APEX_UNUSED(data);
  if (!_terminate) {
      _terminate = true;
      Tau_profile_exit_all_threads();
      Tau_exit("APEX exiting");
  }
  return;
}

void tau_listener::on_new_node(node_event_data &data) {
  if (!_terminate) {
      Tau_set_node(data.node_id);
  }
  return;
}

void tau_listener::on_new_thread(new_thread_event_data &data) {
  if (!_terminate) {
      Tau_register_thread();
      Tau_create_top_level_timer_if_necessary();
      // set the thread id for future listeners to this event
      data.thread_id = Tau_get_thread();
  }
  return;
}

void tau_listener::on_exit_thread(event_data &data) {
  if (!_terminate) {
  }
  APEX_UNUSED(data);
  return;
}

bool tau_listener::on_start(task_identifier * id) {
  if (!_terminate) {
    Tau_start(id->get_name().c_str());
  } else {
      return false;
  }
  return true;
}

bool tau_listener::on_resume(task_identifier * id) {
  return on_start(id);
}

void tau_listener::on_stop(std::shared_ptr<profiler> &p) {
  static string empty("");
  if (!_terminate) {
      if (p->task_id->get_name().compare(empty) == 0) {
          Tau_global_stop(); // stop the top level timer
      } else {
          Tau_stop(p->task_id->get_name().c_str());
      }
  }
  return;
}

void tau_listener::on_yield(std::shared_ptr<profiler> &p) {
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

/* This function is used by APEX threads so that TAU knows about them. */
int initialize_worker_thread_for_TAU(void) {
  if (apex_options::use_tau())
  {
    Tau_register_thread();
    Tau_create_top_level_timer_if_necessary();
  }
  return 0;
}

}

/* Weak symbols that are redefined if we use TAU at runtime */
extern "C" {
APEX_EXPORT int APEX_WEAK Tau_init(int, char**) {
    if (apex::apex_options::use_tau()) {
        /* Print an error message, because TAU wasn't preloaded! */
        std::cerr << "WARNING! TAU libraries not loaded, TAU support unavailable!" << std::endl;
    }
    return 0;
}
APEX_EXPORT int APEX_WEAK initialize_worker_thread_for_tau(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_register_thread(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_create_top_level_timer_if_necessary(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_start(const char *) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_stop(const char *) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_exit(const char*) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_set_node(int) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_profile_exit_all_threads(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_get_thread(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_profile_exit_all_tasks(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_global_stop(void) {return 0;}
APEX_EXPORT int APEX_WEAK Tau_trigger_context_event_thread(char*, double, int) {return 0;}
}
