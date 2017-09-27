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

class message_event_data : public event_data {
public:
  uint64_t tag;
  uint64_t size;
  uint64_t source_rank;
  uint64_t source_thread;
  uint64_t target;
  message_event_data(uint64_t tag, uint64_t size, uint64_t source_rank, uint64_t source_thread, uint64_t target) :
    tag(tag), size(size), source_rank(source_rank), source_thread(source_thread), target(target) {}
  ~message_event_data() {};
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
  uint64_t comm_rank;
  uint64_t comm_size;
  startup_event_data(uint64_t comm_rank, uint64_t comm_size);
};

class dump_event_data : public event_data {
public:
  int node_id;
  bool reset;
  std::string output;
  dump_event_data(int node_id, int thread_id, bool reset);
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
  virtual void on_dump(dump_event_data &data) = 0;
  virtual void on_reset(task_identifier * id) = 0;
  virtual void on_new_node(node_event_data &data) = 0;
  virtual void on_new_thread(new_thread_event_data &data) = 0;
  virtual void on_exit_thread(event_data &data) = 0;
  virtual bool on_start(task_identifier *id) = 0;
  virtual void on_stop(std::shared_ptr<profiler> &p) = 0;
  virtual void on_yield(std::shared_ptr<profiler> &p) = 0;
  virtual bool on_resume(task_identifier * id) = 0;
  virtual void on_new_task(task_identifier * id, uint64_t task_id) = 0;
  virtual void on_sample_value(sample_value_event_data &data) = 0;
  virtual void on_periodic(periodic_event_data &data) = 0;
  virtual void on_custom_event(custom_event_data &data) = 0;
  virtual void on_send(message_event_data &data) = 0;
  virtual void on_recv(message_event_data &data) = 0;
};

}

