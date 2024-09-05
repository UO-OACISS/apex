/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#endif

#include "apex_api.hpp"
#include "profiler.hpp"
#include "task_wrapper.hpp"
#include "event_listener.hpp"
#include "apex_types.h"
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>

#include "profile.hpp"
#include "thread_instance.hpp"
#include <fstream>

// These two are needed by concurrent queue - not defined by Intel Mic support.
#ifndef ATOMIC_BOOL_LOCK_FREE
#define ATOMIC_BOOL_LOCK_FREE      __GCC_ATOMIC_BOOL_LOCK_FREE
#endif
#ifndef ATOMIC_POINTER_LOCK_FREE
#define ATOMIC_POINTER_LOCK_FREE   __GCC_ATOMIC_POINTER_LOCK_FREE
#endif
// HPX has its own version of moodycamel concurrent queue
#ifdef APEX_HAVE_HPX_CONFIG
#include "hpx/concurrency/concurrentqueue.hpp"
using hpx::concurrency::ConcurrentQueue;
#else
#include "concurrentqueue/concurrentqueue.h"
using moodycamel::ConcurrentQueue;
#endif

#include "apex_assert.h"
#include "semaphore.hpp"
#include "task_identifier.hpp"
#include "task_dependency.hpp"
#include <sys/stat.h>
#if !defined(_MSC_VER)
//#include <unistd.h>
//#include <sys/file.h>
#else // not _MSC_VER
#if !defined(__APPLE__)
#include <io.h>
#endif // __APPLE__
#endif // _MSC_VER
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <ctime>

#define INITIAL_NUM_THREADS 2

namespace apex {

class profiler_queue_t : public ConcurrentQueue<std::shared_ptr<profiler> > {
public:
  profiler_queue_t() {}
  virtual ~profiler_queue_t() {
      //finalize();
  }
};

class dependency_queue_t : public ConcurrentQueue<task_dependency*> {
public:
  dependency_queue_t() {}
  virtual ~dependency_queue_t() {
      //finalize();
  }
};

static const char * task_scatterplot_sample_filename = "apex_task_samples.";
static const char * counter_scatterplot_sample_filename = "apex_counter_samples.";

class profiler_listener : public event_listener {
private:
  void _init(void);
  bool _initialized;
  bool _main_timer_stopped;
  std::atomic<bool> _done;
  std::atomic<int> active_tasks;
  std::shared_ptr<profiler> main_timer;
  void write_one_timer(std::string &name, profile * p,
                       std::stringstream &screen_output,
                       double &total_accumulated,
                       double &total_main, double &wall_main, bool include_stops,
                       bool include_papi);
  void finalize_profiles(dump_event_data &data, std::map<std::string, apex_profile*>& profiles);
  void write_taskgraph(void);
  void write_tasktree(void);
  void write_profile(void);
  void delete_profiles(void);
#ifdef APEX_HAVE_HPX
  void schedule_process_profiles(void);
#endif
  unsigned int process_profile(std::shared_ptr<profiler> &p, unsigned int tid);
  unsigned int process_profile(profiler& p, unsigned int tid);
  unsigned int process_dependency(task_dependency* td);
  int node_id;
  std::mutex _mtx;
  bool _common_start(std::shared_ptr<task_wrapper> &tt_ptr,
    bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> &p,
    bool is_yield); // internal, inline function
  void push_profiler(int my_tid, std::shared_ptr<profiler> &p);
  void push_profiler(int my_tid, profiler &p);
  std::unordered_map<task_identifier, profile*> task_map;
  std::mutex _task_map_mutex;
  std::unordered_map<task_identifier, std::unordered_map<task_identifier,
    int>* > task_dependencies;
  /* an vector of profiler queues - so the consumer thread can access them */
  std::mutex queue_mtx;
  std::vector<profiler_queue_t*> allqueues;
  profiler_queue_t * _construct_thequeue(void);
  profiler_queue_t * thequeue(void);
  /* The task dependency queues */
  std::vector<dependency_queue_t*> dependency_queues;
  dependency_queue_t * _construct_dependency_queue(void);
  dependency_queue_t * dependency_queue(void);
  //ConcurrentQueue<task_dependency*> dependency_queue;
  std::unordered_set<task_identifier> throttled_tasks;
  int num_papi_counters;
  std::vector<std::string> metric_names;
#if APEX_HAVE_PAPI
  void initialize_PAPI(bool first_time);
#endif
#ifndef APEX_HAVE_HPX
  std::thread * consumer_thread;
#endif
  semaphore queue_signal;
  std::ofstream _task_scatterplot_sample_file;
  std::ofstream _counter_scatterplot_sample_file;
  std::stringstream task_scatterplot_samples;
  std::stringstream counter_scatterplot_samples;
  std::string timestamp_started;
public:
  void set_node_id(int node_id, int node_count) {
    APEX_UNUSED(node_count);
    this->node_id = node_id;
  }
  profiler_listener (void) : _initialized(false), _main_timer_stopped(false), _done(false),
                             node_id(0), task_map() , num_papi_counters(0),
                             metric_names(0)
  {
      if (apex_options::task_scatterplot()) {
        profiler::get_global_start();
      }
      // get a timestamp for the start of execution
      auto t = std::time(nullptr);
      auto tm = *std::localtime(&t);
      std::ostringstream oss;
      oss << std::put_time(&tm, "%d/%m/%Y %H:%M:%S");
      timestamp_started = oss.str();
  };
  ~profiler_listener (void);
  void async_thread_setup(void);
  // events
  void on_startup(startup_event_data &data);
  void on_dump(dump_event_data &data);
  void on_reset(task_identifier * id);
  void on_pre_shutdown(void);
  void on_shutdown(shutdown_event_data &data);
  void on_new_node(node_event_data &data);
  void on_new_thread(new_thread_event_data &data);
  void on_exit_thread(event_data &data);
  bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
  void on_stop(std::shared_ptr<profiler> &p);
  void on_yield(std::shared_ptr<profiler> &p);
  bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
  void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr);
  void on_sample_value(sample_value_event_data &data);
  void on_periodic(periodic_event_data &data);
  void on_custom_event(custom_event_data &event_data);
  void on_send(message_event_data &data);
  void on_recv(message_event_data &data);
  // other methods
  std::ofstream& task_scatterplot_sample_file() {
      if (!_task_scatterplot_sample_file.is_open()) {
        std::stringstream ss;
        ss << apex_options::output_file_path();
        ss << filesystem_separator();
        ss << task_scatterplot_sample_filename << node_id << ".csv";
        // open the file
        _task_scatterplot_sample_file.open(ss.str(), std::ofstream::out | std::ofstream::app);
        if (!_task_scatterplot_sample_file.is_open()) {
            perror("opening scatterplot sample file");
        }
        APEX_ASSERT(_task_scatterplot_sample_file.is_open());
      }
      return _task_scatterplot_sample_file;
  }
  std::ofstream& counter_scatterplot_sample_file () {
      if (!_counter_scatterplot_sample_file.is_open()) {
        std::stringstream ss;
        ss << apex_options::output_file_path();
        ss << filesystem_separator();
        ss << counter_scatterplot_sample_filename << node_id << ".csv";
        // open the file
        _counter_scatterplot_sample_file.open(ss.str(), std::ofstream::out | std::ofstream::app);
        if (!_counter_scatterplot_sample_file.is_open()) {
            perror("opening scatterplot sample file");
        }
        APEX_ASSERT(_counter_scatterplot_sample_file.is_open());
      }
      return _counter_scatterplot_sample_file;
  }
  void reset(task_identifier * id);
  void reset_all(void);
  profile * get_profile(const task_identifier &id);
  double get_non_idle_time(void);
  profile * get_idle_time(void);
  profile * get_idle_rate(void);
  std::vector<task_identifier>& get_available_profiles() {
    static std::vector<task_identifier> ids;
    std::lock_guard<std::mutex> lock(_task_map_mutex);
    if (task_map.size() > ids.size()) {
        ids.clear();
        for (auto kv : task_map) {
           ids.push_back(kv.first);
        }
    }
    return ids;
  }
  void process_profiles(void);
  static void process_profiles_wrapper(void);
  static void consumer_process_profiles_wrapper(void);
  bool concurrent_cleanup(int i);
#if APEX_HAVE_PAPI
  std::vector<std::string>& get_metric_names(void) { return metric_names; };
#endif
  void stop_main_timer(void);
  void yield_main_timer(void);
  void resume_main_timer(void);
  void increment_main_timer_allocations(double bytes);
  void increment_main_timer_frees(double bytes);
  void push_profiler_public(std::shared_ptr<profiler> &p);
};

}

