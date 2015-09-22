//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROFILER_LISTENER_HPP
#define PROFILER_LISTENER_HPP

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "apex_api.hpp"
#include "event_listener.hpp"
#include "apex_types.h"
#include <boost/atomic.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <boost/thread.hpp>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "profile.hpp"
#include "thread_instance.hpp"
#ifdef USE_UDP
#include "udp_client.hpp"
#endif
#include <boost/functional/hash.hpp>
#include "concurrentqueue/concurrentqueue.h"
#include "semaphore.hpp"

#define INITIAL_NUM_THREADS 2

namespace apex {

class profiler_queue_t : public moodycamel::ConcurrentQueue<profiler*>
{ 
public:
  profiler_queue_t() {}
  virtual ~profiler_queue_t() {
      finalize();
  }
};

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

namespace apex {

class profiler_listener : public event_listener {
private:
  void _init(void);
  boost::atomic<bool> _done;
  boost::atomic<int> active_tasks;
  std::shared_ptr<profiler> main_timer; // not a shared pointer, yet...
  void finalize_profiles(void);
  void write_taskgraph(void);
  void write_profile(int tid);
  void delete_profiles(void);
#ifdef APEX_HAVE_HPX3
  void schedule_process_profiles(void);
#endif
  //unsigned int process_profile(std::shared_ptr<profiler> p, unsigned int tid);
  void reset_all(void);
  unsigned int process_profile(profiler* p, unsigned int tid);
  unsigned int process_dependency(task_dependency* td);
  int node_id;
  boost::mutex _mtx;
  bool _common_start(apex_function_address function_address, bool is_resume); // internal, inline function
  bool _common_start(std::string * timer_name, bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> p, bool is_yield); // internal, inline function
  void push_profiler(int my_tid, std::shared_ptr<profiler> p);
  std::map<std::string, profile*> name_map;
  std::vector<std::map<std::string, profile*>* > thread_name_maps;
  std::map<apex_function_address, profile*> address_map;
  std::vector<std::map<apex_function_address, profile*>* > thread_address_maps;
  std::unordered_map<task_identifier, std::unordered_map<task_identifier, int>* > task_dependencies;
  /* The profiler queue */
  profiler_queue_t thequeue;
  /* The task dependency queue */
  moodycamel::ConcurrentQueue<task_dependency*> dependency_queue;
#if defined(APEX_THROTTLE)
  std::unordered_set<apex_function_address> throttled_addresses;
  std::unordered_set<std::string> throttled_names;
#endif
#if APEX_HAVE_PAPI
  int num_papi_counters;
  std::vector<int> event_sets;
  std::vector<std::string> metric_names;
  void initialize_PAPI(bool first_time);
#endif
  boost::thread * consumer_thread;
  semaphore queue_signal;
public:
  profiler_listener (void) : _done(false), node_id(0), name_map()
#if APEX_HAVE_PAPI
                             , num_papi_counters(0), event_sets(8), metric_names(0)
#endif
  {
#ifdef USE_UDP
      if (apex_options::use_udp_sink()) {
          udp_client::start_client();
      }
#endif
#if APEX_HAVE_PAPI
      num_papi_counters = 0;
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
  void reset(apex_function_address function_address);
  void reset(const std::string &timer_name);
  profile * get_profile(apex_function_address address);
  profile * get_profile(const std::string &timer_name);
  std::vector<std::string> get_available_profiles();
  void process_profiles(void);
  static void process_profiles_wrapper(void);
};

}

#endif // PROFILER_LISTENER_HPP
