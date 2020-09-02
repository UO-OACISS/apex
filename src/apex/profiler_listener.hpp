/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
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
#include <unistd.h>
#include <sys/file.h>
#else // not _MSC_VER
#if !defined(__APPLE__)
#include <io.h>
#endif // __APPLE__
#endif // _MSC_VER
#include <fcntl.h>

#define INITIAL_NUM_THREADS 2

namespace apex {

class profiler_queue_t : public ConcurrentQueue<std::shared_ptr<profiler> > {
public:
  profiler_queue_t() {}
  virtual ~profiler_queue_t() {
      finalize();
  }
};

class dependency_queue_t : public ConcurrentQueue<task_dependency*> {
public:
  dependency_queue_t() {}
  virtual ~dependency_queue_t() {
      finalize();
  }
};

static const char * task_scatterplot_sample_filename = "apex_task_samples.csv";

class profiler_listener : public event_listener {
private:
  void _init(void);
  bool _initialized;
  std::atomic<bool> _done;
  std::atomic<int> active_tasks;
  std::shared_ptr<profiler> main_timer; // not a shared pointer, yet...
  void write_one_timer(task_identifier &task_id, profile * p,
                       std::stringstream &screen_output,
                       std::stringstream &csv_output,
                       double &total_accumulated,
                       double &total_main, bool timer);
  void finalize_profiles(dump_event_data &data);
  void write_taskgraph(void);
  void write_profile(void);
  void delete_profiles(void);
#ifdef APEX_HAVE_HPX
  void schedule_process_profiles(void);
#endif
  unsigned int process_profile(std::shared_ptr<profiler> &p, unsigned int tid);
  unsigned int process_profile(profiler* p, unsigned int tid);
  unsigned int process_dependency(task_dependency* td);
  int node_id;
  std::mutex _mtx;
  bool _common_start(std::shared_ptr<task_wrapper> &tt_ptr,
    bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> &p,
    bool is_yield); // internal, inline function
  void push_profiler(int my_tid, std::shared_ptr<profiler> &p);
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
#if defined(APEX_THROTTLE)
  std::unordered_set<task_identifier> throttled_tasks;
#endif
#if APEX_HAVE_PAPI
  int num_papi_counters;
  std::vector<int> event_sets;
  std::vector<std::string> metric_names;
  void initialize_PAPI(bool first_time);
#endif
#ifndef APEX_HAVE_HPX
  std::thread * consumer_thread;
#endif
  semaphore queue_signal;
  //std::ofstream task_scatterplot_sample_file;
  int task_scatterplot_sample_file;
  std::stringstream task_scatterplot_samples;
public:
  void set_node_id(int node_id, int node_count) {
    APEX_UNUSED(node_count);
    this->node_id = node_id;
  }
  profiler_listener (void) : _initialized(false), _done(false),
                             node_id(0), task_map()
#if APEX_HAVE_PAPI
                             , num_papi_counters(0), event_sets(8),
                             metric_names(0)
#endif
  {
#if APEX_HAVE_PAPI
      num_papi_counters = 0;
#endif
      if (apex_options::task_scatterplot()) {
        // check if the samples file exists
        struct stat buffer;
        std::stringstream ss;
        ss << apex_options::output_file_path();
        ss << filesystem_separator();
        ss << task_scatterplot_sample_filename;
        if (stat (ss.str().c_str(), &buffer) == 0) {
            struct tm *timeinfo = localtime(&buffer.st_mtime);
            time_t filetime = mktime(timeinfo);
            time_t nowish;
            time(&nowish);
            double seconds = difftime(nowish, filetime);
            /* if the file exists, was it recently created? */
            if (seconds > 10) {
                /* create the file */
                std::ofstream tmp;
                tmp.open(ss.str());
                tmp << "#timestamp value   name" << std::endl << std::flush;
                /* yes, close the file because we will use some
                   low-level calls to have concurrent access
                   across processes/threads. */
                tmp.close();
            }
        } else {
            /* create the file */
            std::ofstream tmp;
            tmp.open(ss.str());
            tmp << "#timestamp value   name" << std::endl << std::flush;
            /* yes, close the file because we will use some
            low-level calls to have concurrent access
            across processes/threads. */
            tmp.close();
        }
        // open the file
        task_scatterplot_sample_file = open(ss.str().c_str(), O_APPEND | O_WRONLY );
        if (task_scatterplot_sample_file < 0) {
            perror("opening scatterplot sample file");
        }
        APEX_ASSERT(task_scatterplot_sample_file >= 0);
        profiler::get_global_start();
      }
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
  void reset(task_identifier * id);
  void reset_all(void);
  profile * get_profile(task_identifier &id);
  double get_non_idle_time(void);
  profile * get_idle_time(void);
  profile * get_idle_rate(void);
  //std::vector<std::string> get_available_profiles();
  void process_profiles(void);
  static void process_profiles_wrapper(void);
  static void consumer_process_profiles_wrapper(void);
  bool concurrent_cleanup(int i);
#if APEX_HAVE_PAPI
  std::vector<std::string>& get_metric_names(void) { return metric_names; };
#endif
  void stop_main_timer(void);
  void push_profiler_public(std::shared_ptr<profiler> &p);
};

}

