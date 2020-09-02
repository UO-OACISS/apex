/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#endif

#include <memory>
#include <string>
#include "event_listener.hpp"
#include "thread_instance.hpp"

using namespace std;

namespace apex {

/* this object never actually gets instantiated. too much overhead. */
timer_event_data::timer_event_data(task_identifier * id) : task_id(id) {
  this->my_profiler = std::make_shared<profiler>();
}

/* this object never actually gets instantiated. too much overhead. */
timer_event_data::timer_event_data(std::shared_ptr<profiler> &the_profiler) :
    my_profiler(the_profiler) {
  this->task_id = the_profiler->tt_ptr->get_task_id();
}

timer_event_data::~timer_event_data() {
  //if (have_name)
    //delete(timer_name);
}

node_event_data::node_event_data(int node_id, int thread_id) {
  this->event_type_ = APEX_NEW_NODE;
  this->node_id = node_id;
  this->thread_id = thread_id;
}

sample_value_event_data::sample_value_event_data(int thread_id,
    string counter_name, double counter_value) {
  this->event_type_ = APEX_SAMPLE_VALUE;
  this->is_counter = true;
  this->thread_id = thread_id;
  this->counter_name = new string(counter_name);
  this->counter_value = counter_value;
}

sample_value_event_data::~sample_value_event_data() {
  delete(counter_name);
}

custom_event_data::custom_event_data(apex_event_type event_type,
    void * custom_data) {
    this->event_type_ = event_type;
    this->data = custom_data;
}

custom_event_data::~custom_event_data() {
}

startup_event_data::startup_event_data(uint64_t comm_rank, uint64_t comm_size) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_STARTUP;
  this->comm_rank = comm_rank;
  this->comm_size = comm_size;
}

shutdown_event_data::shutdown_event_data(int node_id, int thread_id) {
  this->event_type_ = APEX_SHUTDOWN;
  this->node_id = node_id;
  this->thread_id = thread_id;
}

dump_event_data::dump_event_data(int node_id, int thread_id, bool reset) {
  this->event_type_ = APEX_DUMP;
  this->node_id = node_id;
  this->thread_id = thread_id;
  this->reset = reset;
  this->output = std::string("");
}

new_thread_event_data::new_thread_event_data(string thread_name) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_THREAD;
  this->thread_name = new string(thread_name);
}

new_thread_event_data::~new_thread_event_data() {
  delete(thread_name);
}

periodic_event_data::periodic_event_data() {
  // don't set the thread ID! It will increment the number of
  // worker threads, and this object is created by the periodic
  // timer thread that is not a worker.
  this->thread_id = 0;
  this->event_type_ = APEX_PERIODIC;
}

}
