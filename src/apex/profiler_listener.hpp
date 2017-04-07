//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef APEX_HAVE_HPX
#include <hpx/config.hpp>
#endif

#include "apex_api.hpp"
#include "event_listener.hpp"
#include "apex_types.h"
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#ifdef APEX_HAVE_HPX
#include <boost/thread.hpp>
#endif

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
#include "concurrentqueue/concurrentqueue.h"

#include "semaphore.hpp"
#include "task_identifier.hpp"
#include "task_dependency.hpp"
#include <sys/stat.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#include <sys/file.h>
#else
#include <io.h>
#endif
#include <fcntl.h>

#define INITIAL_NUM_THREADS 2

// The HPX subsystem creates so many small tasks that it is "better"
// to have a queue per thread, rather than one queue.
//#ifdef APEX_HAVE_HPX
#define APEX_MULTIPLE_QUEUES
//#endif

namespace apex {

class profiler_queue_t : public moodycamel::ConcurrentQueue<std::shared_ptr<profiler> > {
public:
  profiler_queue_t() {}
  virtual ~profiler_queue_t() {
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
                       std::stringstream &screen_output, std::stringstream &csv_output,
                       double &total_accumulated, double &total_main);
  void finalize_profiles(void);
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
  bool _common_start(task_identifier * id, bool is_resume); // internal, inline function
  void _common_stop(std::shared_ptr<profiler> &p, bool is_yield); // internal, inline function
  void push_profiler(int my_tid, std::shared_ptr<profiler> &p);
  std::unordered_map<task_identifier, profile*> task_map;
  std::mutex _task_map_mutex;
  std::unordered_map<task_identifier, std::unordered_map<task_identifier, int>* > task_dependencies;
#ifdef APEX_MULTIPLE_QUEUES
  /* an vector of profiler queues - so the consumer thread can access them */
  std::mutex queue_mtx;
  std::vector<profiler_queue_t*> allqueues;
  profiler_queue_t * _construct_thequeue(void);
  profiler_queue_t * thequeue(void);
#else
  profiler_queue_t thequeue;
#endif
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
#ifdef APEX_HAVE_HPX
  boost::thread * consumer_thread;
#else
  std::thread * consumer_thread;
#endif
  semaphore queue_signal;
  //std::ofstream task_scatterplot_sample_file;
  int task_scatterplot_sample_file;
  std::stringstream task_scatterplot_samples;
public:
  profiler_listener (void) : _initialized(false), _done(false), node_id(0), task_map()
#if APEX_HAVE_PAPI
                             , num_papi_counters(0), event_sets(8), metric_names(0)
#endif
  {
#if APEX_HAVE_PAPI
      num_papi_counters = 0;
#endif
      if (apex_options::task_scatterplot()) {
        // check if the samples file exists
        struct stat buffer;
        if (stat (task_scatterplot_sample_filename, &buffer) == 0) {
            struct tm *timeinfo = localtime(&buffer.st_mtime);
            time_t filetime = mktime(timeinfo);
            time_t nowish;
            time(&nowish);
            double seconds = difftime(nowish, filetime);
            /* if the file exists, was it recently created? */
            if (seconds > 10) {
                /* create the file */
                std::ofstream tmp;
				tmp.open(task_scatterplot_sample_filename);
        		tmp << "#timestamp value   name" << std::endl << std::flush;
				/* yes, close the file because we will use some
				   low-level calls to have concurrent access
                   across processes/threads. */
				tmp.close();
            }
		} else {
            /* create the file */
            std::ofstream tmp;
			tmp.open(task_scatterplot_sample_filename);
        	tmp << "#timestamp value   name" << std::endl << std::flush;
			/* yes, close the file because we will use some
			low-level calls to have concurrent access
            across processes/threads. */
			tmp.close();
        }
		// open the file
		task_scatterplot_sample_file = open(task_scatterplot_sample_filename, O_APPEND | O_WRONLY );
        if (task_scatterplot_sample_file < 0) { perror("opening scatterplot sample file"); }
		assert(task_scatterplot_sample_file >= 0);
        profiler::get_global_start();
      }
  };
  ~profiler_listener (void);
  void async_thread_setup(void);
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
  void on_new_task(task_identifier * id, uint64_t task_id);
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
  void public_process_profile(std::shared_ptr<profiler> &p) { process_profile(p,0); };
  bool concurrent_cleanup(void);
};

}

