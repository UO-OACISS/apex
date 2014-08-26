//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef APEX_EVENTLISTENER_H
#define APEX_EVENTLISTENER_H

#include <string>
#include "apex_types.h"

using namespace std;

namespace apex {

/* Class for holding data relevant to generic event */

class event_data {
public:
  apex_event_type event_type_;
  int thread_id;
  event_data() : thread_id(0) {};
  ~event_data() {};
};

/* Classes for holding data relevant to specific events */

class timer_event_data : public event_data {
public:
  string * timer_name;
  timer_event_data(apex_event_type eventType, int thread_id, string timer_name);
  ~timer_event_data();
};

class node_event_data : public event_data {
public:
  int node_id;
  node_event_data(int node_id, int thread_id);
  ~node_event_data() {};
};

class sample_value_event_data : public event_data {
public:
  string * counter_name;
  double counter_value;
  sample_value_event_data(int thread_id, string counter_name, double counter_value);
  ~sample_value_event_data();
};

class startup_event_data : public event_data {
public:
  int argc;
  char** argv;
  startup_event_data(int argc, char** argv);
  ~startup_event_data() {};
};

class shutdown_event_data : public event_data {
public:
  int node_id;
  shutdown_event_data(int node_id, int thread_id);
  ~shutdown_event_data() {};
};

class new_thread_event_data : public event_data {
public:
  string* thread_name;
  new_thread_event_data(string thread_name);
  ~new_thread_event_data();
};

class periodic_event_data : public event_data {
public:
  periodic_event_data();
  ~periodic_event_data();
};

/* Abstract class for creating an Event Listener class */

class event_listener
{
public:
  // virtual destructor
  virtual ~event_listener() {};
  // all methods in the interface that a handler has to override
  virtual void on_event(event_data * event_data_) = 0;
};

}

#endif // APEX_EVENTLISTENER_H
