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

// These two are needed by concurrent queue - not defined by Intel Mic support.
#ifndef ATOMIC_BOOL_LOCK_FREE
#define ATOMIC_BOOL_LOCK_FREE      __GCC_ATOMIC_BOOL_LOCK_FREE
#endif
#ifndef ATOMIC_POINTER_LOCK_FREE
#define ATOMIC_POINTER_LOCK_FREE   __GCC_ATOMIC_POINTER_LOCK_FREE
#endif
#include "concurrentqueue/concurrentqueue.h"

#include "semaphore.hpp"
#include "task_identifier.hpp"

#define INITIAL_NUM_THREADS 2

namespace apex {

class profiler_queue_t : public moodycamel::ConcurrentQueue<std::shared_ptr<profiler> > { 
public:
  profiler_queue_t() {}
  virtual ~profiler_queue_t() {
      finalize();
  }
};

class profiler_listener : public event_listener {
private:
  void _init(void);
  bool _initialized;
  boost::atomic<bool> _done;
  boost::atomic<int> active_tasks;
  std::shared_ptr<profiler> main_timer; // not a shared pointer, yet...
  void write_one_timer(task_identifier &task_id, profile * p,
                       std::stringstream &screen_output, std::stringstream &csv_output,
                       double &total_accumulated, double &total_main);
  void finalize_profiles(void);
  void write_taskgraph(void);
  void write_profile(void);
  void delete_profiles(void);
#ifdef APEX_HAVE_HPX3
  void schedule_process_profiles(void);
#endif
  unsigned int process_profile(std::shared_ptr<profiler> &p, unsigned int tid);
  unsigned int process_profile(profiler* p, unsigned int tid);
  unsigned int process_dependency(task_dependency* td);
  int node_id;
  boost::mutex _mtx;
  bool _common_start(task_identifier * id, bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> &p, bool is_yield); // internal, inline function
  void push_profiler(int my_tid, std::shared_ptr<profiler> &p);
  std::unordered_map<task_identifier, profile*> task_map;
  std::unordered_map<task_identifier, std::unordered_map<task_identifier, int>* > task_dependencies;
  /* The profiler queue */
  profiler_queue_t thequeue;
  /* The task dependency queue */
  moodycamel::ConcurrentQueue<task_dependency*> dependency_queue;
#if defined(APEX_THROTTLE)
  std::unordered_set<task_identifier> throttled_tasks;
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
  profiler_listener (void) : _initialized(false), _done(false), node_id(0), task_map()
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
  bool on_start(task_identifier *id);
  void on_stop(std::shared_ptr<profiler> &p);
  void on_yield(std::shared_ptr<profiler> &p);
  bool on_resume(task_identifier * id);
  void on_new_task(task_identifier * id, void * task_id);
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &event_data);
  // other methods
  void reset(task_identifier * id);
  void reset_all(void);
  profile * get_profile(task_identifier &id);
  double get_non_idle_time(void);
  profile * get_idle_time(void);
  profile * get_idle_rate(void);
  //std::vector<std::string> get_available_profiles();
  void process_profiles(void);
  static void process_profiles_wrapper(void);
};

}

#endif // PROFILER_LISTENER_HPP
