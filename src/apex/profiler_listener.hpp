//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROFILER_LISTENER_HPP
#define PROFILER_LISTENER_HPP

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "event_listener.hpp"
#include "apex_types.h"
#include <boost/atomic.hpp>
#include <vector>
#include <memory>
#include <boost/thread.hpp>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "profile.hpp"
#include "thread_instance.hpp"
#ifdef USE_UDP
#include "udp_client.hpp"
#endif
#include <boost/functional/hash.hpp>



namespace apex {

class task_identifier {
public:
  apex_function_address address;
  std::string name;
  std::string _resolved_name;
  bool has_name;
  task_identifier(apex_function_address a) : 
      address(a), name(""), _resolved_name(""), has_name(false) {};
  task_identifier(std::string n) : 
      address(0L), name(n), _resolved_name(""), has_name(true) {};
  task_identifier(profiler * p) : 
      address(0L), name(""), _resolved_name("") {
      if (p->have_name) {                                         
          name = *p->timer_name;
          has_name = true;
      } else {                                                         
          address = p->action_address;
          has_name = false;
      }            
  }
  std::string& get_name() {
    if (!has_name) {
      if (_resolved_name == "") {
        //_resolved_name = lookup_address((uintptr_t)address, false);         
        _resolved_name = thread_instance::instance().map_addr_to_name(address);
      }
      return _resolved_name;
    }
    return name;
  }
  ~task_identifier() { }
  // requried for using this class as a key in an unordered map.
  // the hash function is defined below.
  bool operator==(const task_identifier &other) const { 
    return (address == other.address && name.compare(other.name) == 0);
  }
};

class task_dependency {
public:
  task_identifier * parent;
  task_identifier * child;
  task_dependency(task_identifier * p, task_identifier * c) :
    parent(p), child(c) {};
  ~task_dependency() {
    delete parent;
    delete child;
  }
};

class profiler_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
  static boost::atomic<int> active_tasks;
  static std::shared_ptr<profiler> main_timer; // not a shared pointer, yet...
  static void finalize_profiles(void);
  static void write_taskgraph(void);
  static void write_profile(int tid);
  static void delete_profiles(void);
#ifdef APEX_HAVE_HPX3
  static void schedule_process_profiles(void);
#endif
  //static unsigned int process_profile(std::shared_ptr<profiler> p, unsigned int tid);
  static unsigned int process_profile(profiler* p, unsigned int tid);
  static unsigned int process_dependency(task_dependency* td);
  static int node_id;
  static boost::mutex _mtx;
  bool _common_start(apex_function_address function_address, bool is_resume); // internal, inline function
  bool _common_start(std::string * timer_name, bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> p, bool is_yield); // internal, inline function
  static void push_profiler(int my_tid, std::shared_ptr<profiler> p);
public:
  profiler_listener (void)  : _terminate(false) {
#ifdef USE_UDP
      if (apex_options::use_udp_sink()) {
          udp_client::start_client();
      }
#endif
  };
  ~profiler_listener (void);
  // events
  void on_startup(startup_event_data &data);
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_exit_thread(event_data &data);
  bool on_start(apex_function_address function_address);
  bool on_start(std::string *timer_name);
  void on_stop(std::shared_ptr<profiler> p);
  void on_yield(std::shared_ptr<profiler> p);
  bool on_resume(apex_function_address function_address);
  bool on_resume(std::string *timer_name);
  void on_new_task(apex_function_address function_address, void * task_id);
  void on_new_task(std::string *timer_name, void * task_id);
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &event_data);
  // other methods
  static void reset(apex_function_address function_address);
  static void reset(const std::string &timer_name);
  static profile * get_profile(apex_function_address address);
  static profile * get_profile(const std::string &timer_name);
  static std::vector<std::string> get_available_profiles();
  static void process_profiles(void);
};

}

/* This is the hash function for the task_identifier class */
namespace std {

  template <>
  struct hash<apex::task_identifier>
  {
    std::size_t operator()(const apex::task_identifier& k) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed,boost::hash_value(k.address));
      boost::hash_combine(seed,boost::hash_value(k.name));
      return seed;
    }
  };

}

#endif // PROFILER_LISTENER_HPP
