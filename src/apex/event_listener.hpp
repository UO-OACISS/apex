//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <string>
#include <memory>
#include "apex_types.h"
#include "profiler.hpp"
#include "task_identifier.hpp"

namespace apex {

/* Class for holding data relevant to generic event */

class event_data {
public:
  apex_event_type event_type_;
  int thread_id;
  void * data; /* generic data pointer */
  event_data() : thread_id(0), data(NULL) {};
  virtual ~event_data() {};
};

/* Classes for holding data relevant to specific events */

class timer_event_data : public event_data {
public:
  task_identifier * task_id;
  std::shared_ptr<profiler> my_profiler;
  timer_event_data(task_identifier * id);
  timer_event_data(std::shared_ptr<profiler> &the_profiler);
  ~timer_event_data();
};

class node_event_data : public event_data {
public:
  int node_id;
  node_event_data(int node_id, int thread_id);
};

class sample_value_event_data : public event_data {
public:
  std::string * counter_name;
  double counter_value;
  bool is_counter;
  sample_value_event_data(int thread_id, std::string counter_name, double counter_value);
  ~sample_value_event_data();
};

class startup_event_data : public event_data {
public:
  int argc;
  char** argv;
  startup_event_data(int argc, char** argv);
};

class shutdown_event_data : public event_data {
public:
  int node_id;
  shutdown_event_data(int node_id, int thread_id);
};

class new_thread_event_data : public event_data {
public:
  std::string* thread_name;
  new_thread_event_data(std::string thread_name);
  ~new_thread_event_data();
};

class new_task_event_data : public event_data {
public:
  task_identifier * task_id;
  new_task_event_data(task_identifier * task_id, void * data);
  ~new_task_event_data();
};

class destroy_task_event_data : public event_data {
public:
  task_identifier * task_id;
  destroy_task_event_data(task_identifier * task_id, void * data);
  ~destroy_task_event_data();
};

class new_dependency_event_data : public event_data {
public:
  task_identifier * src;
  task_identifier * dest;
  new_dependency_event_data(task_identifier * src, task_identifier * dest);
  ~new_dependency_event_data();
};

class satisfy_dependency_event_data : public event_data {
public:
  task_identifier * src;
  task_identifier * dest;
  satisfy_dependency_event_data(task_identifier * src, task_identifier * dest);
  ~satisfy_dependency_event_data();
};

class set_task_state_event_data : public event_data {
  task_identifier * task_id;
  apex_task_state   state;
  set_task_state_event_data(task_identifier * task_id, apex_task_state state);
  ~set_task_state_event_data();
};

class acquire_data_event_data : public event_data {
public:
  task_identifier * task_id;
  task_identifier * data_id;
  uint64_t size;
  acquire_data_event_data(task_identifier * task_id, task_identifier * dest, uint64_t size);
  ~acquire_data_event_data();
};

class release_data_event_data : public event_data {
public:
  task_identifier * task_id;
  task_identifier * data_id;
  uint64_t size;
  release_data_event_data(task_identifier * task_id, task_identifier * dest, uint64_t size);
  ~release_data_event_data();
};

class new_event_event_data : public event_data {
public:
  task_identifier * event_id;
  new_event_event_data(task_identifier * event_id);
  ~new_event_event_data();
};

class destroy_event_event_data : public event_data {
public:
  task_identifier * event_id;
  destroy_event_event_data(task_identifier * event_id);
  ~destroy_event_event_data();
};

class new_data_event_data : public event_data {
public:
  task_identifier * data_id;
  uint64_t size;
  new_data_event_data(task_identifier * data_id, uint64_t size);
  ~new_data_event_data();
};

class destroy_data_event_data : public event_data {
public:
  task_identifier * data_id;
  destroy_data_event_data(task_identifier * data_id);
  ~destroy_data_event_data();
};

class periodic_event_data : public event_data {
public:
  periodic_event_data();
};

class custom_event_data : public event_data {
public:
  custom_event_data(apex_event_type event_type, void * custom_data);
  ~custom_event_data();
};

/* Abstract class for creating an Event Listener class */

class event_listener
{
public:
  // virtual destructor
  virtual ~event_listener() {};
  // all methods in the interface that a handler has to override
  virtual void on_startup(startup_event_data &data) = 0;
  virtual void on_shutdown(shutdown_event_data &data) = 0;
  virtual void on_new_node(node_event_data &data) = 0;
  virtual void on_new_thread(new_thread_event_data &data) = 0;
  virtual void on_exit_thread(event_data &data) = 0;
  virtual bool on_start(task_identifier *id) = 0;
  virtual void on_stop(std::shared_ptr<profiler> &p) = 0;
  virtual void on_yield(std::shared_ptr<profiler> &p) = 0;
  virtual bool on_resume(task_identifier * id) = 0;
  virtual void on_new_task(new_task_event_data &data) = 0;
  virtual void on_destroy_task(destroy_task_event_data &data) = 0;
  virtual void on_new_dependency(new_dependency_event_data &data) = 0;
  virtual void on_satisfy_dependency(satisfy_dependency_event_data &data) = 0;
  virtual void on_set_task_state(set_task_state_event_data &data) = 0;
  virtual void on_acquire_data(acquire_data_event_data &data) = 0;
  virtual void on_release_data(release_data_event_data &data) = 0;
  virtual void on_new_event(new_event_event_data &data) = 0;
  virtual void on_destroy_event(destroy_event_event_data &data) = 0;
  virtual void on_new_data(new_data_event_data &data) = 0;
  virtual void on_destroy_data(destroy_data_event_data &data) = 0;
  virtual void on_sample_value(sample_value_event_data &data) = 0;
  virtual void on_periodic(periodic_event_data &data) = 0;
  virtual void on_custom_event(custom_event_data &data) = 0;
};

}

