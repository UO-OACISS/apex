//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "event_listener.hpp"
#include "thread_instance.hpp"

/* At some point, make this multithreaded using the multiproducer/singlecomsumer example
 * at http://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html
 */

using namespace std;

namespace apex {

timer_event_data::timer_event_data(const string &timer_name) : have_name(true) {
  this->start_timestamp = std::chrono::high_resolution_clock::now();
  this->my_profiler = NULL;
  this->timer_name = new string(timer_name);
  this->function_address = APEX_NULL_FUNCTION_ADDRESS;
}

timer_event_data::timer_event_data(apex_function_address function_address) : have_name(false) {
  this->start_timestamp = std::chrono::high_resolution_clock::now();
  this->my_profiler = NULL;
  this->function_address = function_address;
}

timer_event_data::timer_event_data(profiler * the_profiler) : have_name(false) {
  this->start_timestamp = the_profiler->start;
  this->end_timestamp = std::chrono::high_resolution_clock::now();
  this->my_profiler = the_profiler;
  if (the_profiler->have_name) {
    this->have_name = true;
    this->timer_name = the_profiler->timer_name;
    this->function_address = APEX_NULL_FUNCTION_ADDRESS;
  } else {
    this->timer_name = NULL;
    this->function_address = the_profiler->action_address;
  }
}

timer_event_data::~timer_event_data() {
  //if (have_name)
    //delete(timer_name);
}

node_event_data::node_event_data(int node_id, int thread_id) {
  //this->event_type_ = APEX_NEW_NODE;
  this->node_id = node_id;
  this->thread_id = thread_id;
}

sample_value_event_data::sample_value_event_data(int thread_id, string counter_name, double counter_value) {
  //this->event_type_ = APEX_SAMPLE_VALUE;
  this->is_counter = true;
  this->thread_id = thread_id;
  this->counter_name = new string(counter_name);
  this->counter_value = counter_value;
}

sample_value_event_data::~sample_value_event_data() {
  delete(counter_name);
}

custom_event_data::custom_event_data(apex_event_type event_type, void * custom_data) {
    this->event_type_ = event_type;
    this->data = custom_data;
}

custom_event_data::~custom_event_data() {
}

startup_event_data::startup_event_data(int argc, char** argv) {
  this->thread_id = thread_instance::get_id();
  //this->event_type_ = APEX_STARTUP;
  this->argc = argc;
  this->argv = argv;
}

shutdown_event_data::shutdown_event_data(int node_id, int thread_id) {
  //this->event_type_ = APEX_SHUTDOWN;
  this->node_id = node_id;
  this->thread_id = thread_id;
}

new_thread_event_data::new_thread_event_data(string thread_name) {
  this->thread_id = thread_instance::get_id();
  //this->event_type_ = APEX_NEW_THREAD;
  this->thread_name = new string(thread_name);
}

new_thread_event_data::~new_thread_event_data() {
  delete(thread_name);
}

periodic_event_data::periodic_event_data() {
  this->thread_id = thread_instance::get_id();
  //this->event_type_ = APEX_PERIODIC;
}

}
