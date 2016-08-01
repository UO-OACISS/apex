//  Copyright (c) 2014-2016 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "event_listener.hpp"
#include "thread_instance.hpp"

using namespace std;

namespace apex {

/* this object never actually gets instantiated. too much overhead. */
timer_event_data::timer_event_data(task_identifier * id) : task_id(id) {
  this->my_profiler = std::make_shared<profiler>();
}

/* this object never actually gets instantiated. too much overhead. */
timer_event_data::timer_event_data(std::shared_ptr<profiler> &the_profiler) : my_profiler(the_profiler) {
  this->task_id = the_profiler->task_id; 
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

sample_value_event_data::sample_value_event_data(int thread_id, string counter_name, double counter_value) {
  this->event_type_ = APEX_SAMPLE_VALUE;
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
  this->event_type_ = APEX_STARTUP;
  this->argc = argc;
  this->argv = argv;
}

shutdown_event_data::shutdown_event_data(int node_id, int thread_id) {
  this->event_type_ = APEX_SHUTDOWN;
  this->node_id = node_id;
  this->thread_id = thread_id;
}

new_thread_event_data::new_thread_event_data(string thread_name) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_THREAD;
  this->thread_name = new string(thread_name);
}

new_thread_event_data::~new_thread_event_data() {
  delete(thread_name);
}

new_task_event_data::new_task_event_data(task_identifier * task_id, void * data) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_TASK;
  this->task_id = task_id;
  this->data = data;
}

new_task_event_data::~new_task_event_data() {
}

destroy_task_event_data::destroy_task_event_data(task_identifier * task_id, void * data) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_DESTROY_TASK;
  this->task_id = task_id;
  this->data = data;
}

destroy_task_event_data::~destroy_task_event_data() {
}

new_dependency_event_data::new_dependency_event_data(task_identifier * src, task_identifier * dest) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_DEPENDENCY;
  this->src = src;
  this->dest = dest;
}

new_dependency_event_data::~new_dependency_event_data() {
}


satisfy_dependency_event_data::satisfy_dependency_event_data(task_identifier * src, task_identifier * dest) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_SATISFY_DEPENDENCY;
  this->src = src;
  this->dest = dest;
}

satisfy_dependency_event_data::~satisfy_dependency_event_data() {
}

acquire_data_event_data::acquire_data_event_data(task_identifier * task_id,
        task_identifier * data_id, uint64_t size) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_ACQUIRE_DATA;
  this->task_id = task_id;
  this->data_id = data_id;
  this->size = size;
}

acquire_data_event_data::~acquire_data_event_data() {
}

release_data_event_data::release_data_event_data(task_identifier * task_id,
        task_identifier * dest, uint64_t size) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_RELEASE_DATA;
  this->task_id = task_id;
  this->data_id = data_id;
  this->size = size;
}                         

release_data_event_data::~release_data_event_data() {
}

new_event_event_data::new_event_event_data(task_identifier * event_id) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_EVENT;
  this->event_id = event_id;
}

new_event_event_data::~new_event_event_data() {
}

destroy_event_event_data::destroy_event_event_data(task_identifier * event_id) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_DESTROY_EVENT;
  this->event_id = event_id;
}

destroy_event_event_data::~destroy_event_event_data() {
}

new_data_event_data::new_data_event_data(task_identifier * data_id) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_NEW_DATA;
  this->data_id = data_id;
}

new_data_event_data::~new_data_event_data() {
}

destroy_data_event_data::destroy_data_event_data(task_identifier * data_id) {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_DESTROY_DATA;
  this->data_id = data_id;
}

destroy_data_event_data::~destroy_data_event_data() {
}

set_task_state_event_data::set_task_state_event_data(task_identifier * task_id, apex_task_state state) {
    this->thread_id = thread_instance::get_id();
    this->event_type_ = APEX_SET_TASK_STATE;
    this->task_id = task_id;
    this->state = state;
}

set_task_state_event_data::~set_task_state_event_data() {
}

periodic_event_data::periodic_event_data() {
  this->thread_id = thread_instance::get_id();
  this->event_type_ = APEX_PERIODIC;
}

}
