//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_HPP
#define APEX_HPP

#include <string>
#include <vector>
#include <stdint.h>
#include "handler.hpp"
#include "event_listener.hpp"
#include "policy_handler.hpp"

//using namespace std;

namespace apex {

class apex {
private:
// private constructors cannot be called
  apex() : m_argc(0), m_argv(NULL), m_node_id(0) {_initialize();};
  apex(int argc, char**argv) : m_argc(argc), m_argv(argv) {_initialize();};
  apex(apex const&){};             // copy constructor is private
  apex& operator=(apex const& a){ return const_cast<apex&>(a); };  // assignment operator is private
  static apex* m_pInstance;
  int m_argc;
  char** m_argv;
  int m_node_id;
  bool m_profiling;
  void _initialize();
  policy_handler * m_policy_handler;
  std::map<int, policy_handler*> period_handlers;
  std::vector<event_listener*> listeners;
public:
  string* m_my_locality;
  static apex* instance(); // singleton instance
  static apex* instance(int argc, char** argv); // singleton instance
  void set_node_id(int id);
  int get_node_id(void);
  void notify_listeners(event_data* event_data_);
  policy_handler * get_policy_handler(void) const;
  policy_handler * get_policy_handler(int period);
  ~apex();
};

void init(void);
void init(int argc, char** argv);
void finalize(void);
double version(void);
void start(std::string timer_name);
void stop(std::string timer_name);
void stop(void);
void sample_value(std::string name, double value);
void set_node_id(int id);
void register_thread(std::string name);
void track_power(void);
void track_power_here(void);
void enable_tracking_power(void);
void disable_tracking_power(void);
void set_interrupt_interval(int seconds);
int register_event_policy(const std::set<_event_type> & when,
  bool (*test_function)(void* arg1), void (*action_function)(void* arg2));
int register_periodic_policy(int period, bool (*test_function)(void* arg1),
  void (*action_function)(void* arg2));
}

#endif //APEX_HPP
