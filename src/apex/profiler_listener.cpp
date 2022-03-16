/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#ifdef APEX_HAVE_OTF2
#define APEX_TRACE_APEX
#endif // APEX_HAVE_OTF2
#endif // APEX_HAVE_HPX_CONFIG

#include "profiler_listener.hpp"
#include "profiler.hpp"
#include "task_wrapper.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "profile.hpp"
#include "apex.hpp"

#include <atomic>
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || \
    (defined(__APPLE__) && defined(__MACH__)))
#include <unistd.h>
#include <sched.h>
#endif
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <regex>

#include <functional>
#include <thread>
#include <future>

#if defined(APEX_THROTTLE)
#include "apex_cxx_shared_lock.hpp"
apex::shared_mutex_type throttled_event_set_mutex;
#define APEX_THROTTLE_CALLS 1000
#endif

#if APEX_HAVE_PAPI
#include "papi.h"
#include <mutex>
std::mutex event_set_mutex;
#endif

#ifdef APEX_HAVE_HPX
#include <boost/assign.hpp>
#include <cstdint>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos_local/composable_guard.hpp>
static void apex_schedule_process_profiles(void); // not in apex namespace
const int num_non_worker_threads_registered = 1; // including the main thread
//bool synchronous_flush{false};
#endif

#define APEX_SYNCHRONOUS_PROCESSING 1

#include "tau_listener.hpp"
#include "utils.hpp"
#include "profile_reducer.hpp"

#include <cstdlib>
#include <ctime>

#include <iomanip>
#include <locale>

using namespace std;
using namespace apex;

enum papi_state { papi_running, papi_suspended };
class profiler_listener_globals {
public:
    unsigned int my_tid; // the current thread's TID in APEX
    std::vector<int> event_sets; // PAPI event sets
    std::vector<size_t> event_set_sizes; // PAPI event set sizes
    papi_state thread_papi_state;
    profiler_listener_globals() : my_tid(-1), thread_papi_state(papi_suspended) { }
    ~profiler_listener_globals() { if (my_tid == 0) finalize(); }
};

APEX_NATIVE_TLS profiler_listener_globals _pls;

namespace apex {

/* mutex for controlling access to the dependencies map.
 * multiple threads can try to clean up at the same time. */
std::mutex task_dependency_mutex;
/* set for keeping track of memory to clean up */
std::mutex free_profile_set_mutex;
std::unordered_set<profile*> free_profiles;

    /* We do this in two stages, to make the common case fast. */
    profiler_queue_t * profiler_listener::_construct_thequeue() {
        profiler_queue_t * _thequeue = new profiler_queue_t();
        /* We are locking to make sure the vector is only updated by
         * one thread at a time. */
        std::unique_lock<std::mutex> queue_lock(queue_mtx);
        allqueues.push_back(_thequeue);
        return _thequeue;
    }
    /* this is a thread-local pointer to a concurrent queue for each worker thread. */
    profiler_queue_t * profiler_listener::thequeue() {
        /* This constructor gets called once per thread, the first time this
         * function is executed (by each thread). */
        static APEX_NATIVE_TLS profiler_queue_t * _thequeue = _construct_thequeue();
        return _thequeue;
    }

    /* We do this in two stages, to make the common case fast. */
    dependency_queue_t * profiler_listener::_construct_dependency_queue() {
        dependency_queue_t * _thequeue = new dependency_queue_t();
        /* We are locking to make sure the vector is only updated by
         * one thread at a time. */
        std::unique_lock<std::mutex> queue_lock(queue_mtx);
        dependency_queues.push_back(_thequeue);
        return _thequeue;
    }
    /* this is a thread-local pointer to a concurrent queue for each worker thread. */
    dependency_queue_t * profiler_listener::dependency_queue() {
        /* This constructor gets called once per thread, the first time this
         * function is executed (by each thread). */
        static APEX_NATIVE_TLS dependency_queue_t * _thequeue =
            _construct_dependency_queue();
        return _thequeue;
    }

  /* Flag indicating whether a consumer task is currently running */
  std::atomic_flag consumer_task_running = ATOMIC_FLAG_INIT;
#ifdef APEX_HAVE_HPX
  bool hpx_shutdown = false;
#endif

  double profiler_listener::get_non_idle_time() {
    double non_idle_time = 0.0;
    /* Iterate over all timers and accumulate the time spent in them */
    unordered_map<task_identifier, profile*>::const_iterator it2;
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
      profile * p = it2->second;
#if defined(APEX_THROTTLE)
      if (!apex_options::use_tau()) {
        task_identifier id = it2->first;
        unordered_set<task_identifier>::const_iterator it4;
        {
              read_lock_type l(throttled_event_set_mutex);
              it4 = throttled_tasks.find(id);
        }
        if (it4!= throttled_tasks.end()) {
            continue;
        }
      }
#endif
      if (p->get_type() == APEX_TIMER) {
        non_idle_time += p->get_accumulated();
      }
    }
    return non_idle_time;
  }

  profile * profiler_listener::get_idle_time() {
    double non_idle_time = get_non_idle_time();
    /* Subtract the accumulated time from the main time span. */
    int num_worker_threads = thread_instance::get_num_workers();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_main = main_timer->elapsed() *
                fmin(hardware_concurrency(), num_worker_threads);
    double elapsed = total_main - non_idle_time;
    elapsed = elapsed > 0.0 ? elapsed : 0.0;
    profile * theprofile = new profile(elapsed, 0, nullptr, false);
    {
        std::unique_lock<std::mutex> l(free_profile_set_mutex);
        free_profiles.insert(theprofile);
    }
    return theprofile;
  }

  profile * profiler_listener::get_idle_rate() {
    double non_idle_time = get_non_idle_time();
    /* Subtract the accumulated time from the main time span. */
    int num_worker_threads = thread_instance::get_num_workers();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_main = main_timer->elapsed() *
                fmin(hardware_concurrency(), num_worker_threads);
    double elapsed = total_main - non_idle_time;
    double rate = elapsed > 0.0 ? ((elapsed/total_main)) : 0.0;
    profile * theprofile = new profile(rate, 0, nullptr, false);
    {
        std::unique_lock<std::mutex> l(free_profile_set_mutex);
        free_profiles.insert(theprofile);
    }
    return theprofile;
  }

  /* Return the requested profile object to the user.
   * Return nullptr if doesn't exist. */
  profile * profiler_listener::get_profile(const task_identifier &id) {
    /* Maybe we aren't processing profiler objects yet? Fire off a request. */
#ifndef APEX_SYNCHRONOUS_PROCESSING
#ifdef APEX_HAVE_HPX
    // don't schedule an HPX action - just do it.
    process_profiles_wrapper();
#else
    queue_signal.post();
#endif
#endif // APEX_SYNCHRONOUS_PROCESSING
    if (id.name == string(APEX_IDLE_RATE)) {
        return get_idle_rate();
    } else if (id.name == string(APEX_IDLE_TIME)) {
        return get_idle_time();
    } else if (id.name == string(APEX_NON_IDLE_TIME)) {
        profile * theprofile = new profile(get_non_idle_time(), 0, nullptr, false);
        {
            std::unique_lock<std::mutex> l(free_profile_set_mutex);
            free_profiles.insert(theprofile);
        }
        return theprofile;
    }
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    unordered_map<task_identifier, profile*>::const_iterator it = task_map.find(id);
    if (it != task_map.end()) {
      return (*it).second;
    }
    return nullptr;
  }

  void profiler_listener::reset_all(void) {
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(auto &it : task_map) {
        it.second->reset();
    }
    if (apex_options::use_jupyter_support()) {
        // restart the main timer
        main_timer = std::make_shared<profiler>(task_wrapper::get_apex_main_wrapper());
    }
  }

  /* After the consumer thread pulls a profiler off of the queue,
   * process it by updating its profile object in the map of profiles. */
  // TODO The name-based timer and address-based timer paths through
  // the code involve a lot of duplication -- this should be refactored
  // to remove the duplication so it's easier to maintain.
  unsigned int profiler_listener::process_profile(
    std::shared_ptr<profiler> &p, unsigned int tid)
  {
    if(p == nullptr) return 0;
    return process_profile(*p,tid);
  }

  unsigned int profiler_listener::process_profile(profiler& p, unsigned int tid)
  {
    APEX_UNUSED(tid);
    profile * theprofile;
    if(p.is_reset == reset_type::ALL) {
        reset_all();
        return 0;
    }
    double values[8] = {0};
    double tmp_num_counters = 0;
#if APEX_HAVE_PAPI
    tmp_num_counters = num_papi_counters;
    for (int i = 0 ; i < num_papi_counters ; i++) {
        if (p.papi_stop_values[i] > p.papi_start_values[i]) {
            values[i] = p.papi_stop_values[i] - p.papi_start_values[i];
        } else {
            values[i] = 0.0;
        }
    }
#endif
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    unordered_map<task_identifier, profile*>::const_iterator it =
        task_map.find(*(p.get_task_id()));
    if (it != task_map.end()) {
          // A profile for this ID already exists.
        theprofile = (*it).second;
        task_map_lock.unlock();
        if(p.is_reset == reset_type::CURRENT) {
            theprofile->reset();
        } else {
            if (apex_options::track_memory()) {
                theprofile->increment(p.elapsed(), tmp_num_counters,
                    values, p.allocations, p.frees, p.bytes_allocated,
                    p.bytes_freed, p.is_resume);
            } else {
                theprofile->increment(p.elapsed(), tmp_num_counters,
                    values, p.is_resume);
            }
        }
#if defined(APEX_THROTTLE)
        if (!apex_options::use_tau()) {
          // Is this a lightweight task? If so, we shouldn't measure it any more,
          // in order to reduce overhead.
          if (theprofile->get_calls() > APEX_THROTTLE_CALLS &&
              theprofile->get_mean() < APEX_THROTTLE_PERCALL) {
              unordered_set<task_identifier>::const_iterator it2;
              {
                  read_lock_type l(throttled_event_set_mutex);
                it2 = throttled_tasks.find(*(p.get_task_id()));
              }
              if (it2 == throttled_tasks.end()) {
                  // lock the set for insert
                  {
                        write_lock_type l(throttled_event_set_mutex);
                      // was it inserted when we were waiting?
                      it2 = throttled_tasks.find(*(p.get_task_id()));
                      // no? OK - insert it.
                      if (it2 == throttled_tasks.end()) {
                          throttled_tasks.insert(*(p.get_task_id()));
                      }
                  }
                  if (apex_options::use_verbose()) {
                      cout << "APEX: disabling lightweight timer "
                           << p.get_task_id()->get_name()
                            << endl;
                      fflush(stdout);
                  }
              }
          }
        }
#endif
      } else {
        // Create a new profile for this name.
        if (apex_options::track_memory() && !p.is_counter) {
            theprofile = new profile(p.is_reset ==
                reset_type::CURRENT ? 0.0 : p.elapsed(),
                tmp_num_counters, values, p.is_resume,
                p.allocations, p.frees, p.bytes_allocated,
                p.bytes_freed);
            task_map[*(p.get_task_id())] = theprofile;
        } else {
            theprofile = new profile(p.is_reset ==
                reset_type::CURRENT ? 0.0 : p.elapsed(),
                tmp_num_counters, values, p.is_resume,
                p.is_counter ? APEX_COUNTER : APEX_TIMER);
            task_map[*(p.get_task_id())] = theprofile;
        }
        task_map_lock.unlock();
#ifdef APEX_HAVE_HPX
#ifdef APEX_REGISTER_HPX3_COUNTERS
        if(!_done) {
            if(get_hpx_runtime_ptr() != nullptr &&
                p.get_task_id()->has_name()) {
                std::string timer_name(p.get_task_id()->get_name());
                //Don't register timers containing "/"
                if(timer_name.find("/") == std::string::npos) {
                    hpx::performance_counters::install_counter_type(
                    std::string("/apex/") + timer_name,
                    [p](bool r)->std::int64_t{
                        std::int64_t value(p.elapsed());
                        return value;
                    },
                    std::string("APEX counter ") + timer_name,
                    ""
                    );
                }
            } else {
                std::cerr << "HPX runtime not initialized yet." << std::endl;
            }
        }
#endif
#endif
      }
      /* write the sample to the file */
      if (apex_options::task_scatterplot()) {
        if (!p.is_counter) {
            static int thresh = std::round((double)(RAND_MAX) * apex_options::scatterplot_fraction());
            if (std::rand() < thresh) {
                /* before calling p.get_task_id()->get_name(), make sure we create
                 * a thread_instance object that is NOT a worker. */
                thread_instance::instance(false);
                std::unique_lock<std::mutex> task_map_lock(_mtx);
                task_scatterplot_samples << p.normalized_timestamp() << " "
                            << p.elapsed() << " "
                            << "'" << p.get_task_id()->get_name() << "'" << endl;
                int loc0 = task_scatterplot_samples.tellp();
                if (loc0 > 32768) {
                    task_scatterplot_sample_file() << task_scatterplot_samples.rdbuf();
                    // reset the stringstream
                    task_scatterplot_samples.str("");
                }
            }
        } else {
                thread_instance::instance(false);
                std::unique_lock<std::mutex> task_map_lock(_mtx);
                counter_scatterplot_samples << p.normalized_timestamp() << " "
                            << p.elapsed() << " "
                            << "'" << p.get_task_id()->get_name() << "'" << endl;
                int loc0 = task_scatterplot_samples.tellp();
                if (loc0 > 32768) {
                    counter_scatterplot_sample_file() << counter_scatterplot_samples.rdbuf();
                    // reset the stringstream
                    counter_scatterplot_samples.str("");
                }
	}
      }
    if (apex_options::use_tasktree_output() && !p.is_counter && p.tt_ptr != nullptr) {
        p.tt_ptr->tree_node->addAccumulated(p.elapsed_seconds(), p.is_resume);
    }
    return 1;
  }

  inline unsigned int profiler_listener::process_dependency(task_dependency* td)
  {
      unordered_map<task_identifier,
        unordered_map<task_identifier,
        int>* >::const_iterator it = task_dependencies.find(td->parent);
      unordered_map<task_identifier, int> * depend;
      // if this is a new dependency for this parent?
      if (it == task_dependencies.end()) {
          depend = new unordered_map<task_identifier, int>();
          (*depend)[td->child] = 1;
          task_dependencies[td->parent] = depend;
      // otherwise, see if this parent has seen this child
      } else {
          depend = it->second;
          unordered_map<task_identifier, int>::const_iterator it2 =
            depend->find(td->child);
          // first time for this child
          if (it2 == depend->end()) {
              (*depend)[td->child] = 1;
          // not the first time for this child
          } else {
              int tmp = it2->second;
              (*depend)[td->child] = tmp + 1;
          }
      }
      delete(td);
      return 1;
  }

  /* Cleaning up memory. Not really necessary, because it only gets
   * called at shutdown. But a good idea to do regardless. */
  void profiler_listener::delete_profiles(void) {
    // iterate over the map and free the objects in the map
    unordered_map<task_identifier, profile*>::const_iterator it;
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it = task_map.begin(); it != task_map.end(); it++) {
      delete it->second;
    }
    // clear the map.
    task_map.clear();

  }

#define PAD_WITH_SPACES "%8s"
#define FORMAT_PERCENT "%8.3f"
#define FORMAT_SCIENTIFIC "%1.2e"

  template<typename ... Args>
  string string_format( const std::string& format, Args ... args )
  {
      // Extra space for '\0'
      size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1;
      unique_ptr<char[]> buf( new char[ size ] );
      snprintf( buf.get(), size, format.c_str(), args ... );
      // We don't want the '\0' inside
      return string( buf.get(), buf.get() + size - 1 );
  }

  void profiler_listener::write_one_timer(std::string &action_name,
          profile * p, stringstream &screen_output,
          stringstream &csv_output, double &total_accumulated,
          double &total_main, bool timer, bool include_stops = false,
          bool include_papi = false) {
      string shorter(action_name);
      size_t maxlength = 41;
      if (timer) maxlength = 52;
      // to keep formatting pretty, trim any long timer names
      if (shorter.size() > maxlength) {
        shorter.resize(maxlength-3);
        shorter.resize(maxlength, '.');
      }
      //screen_output << "\"" << shorter << "\", " ;
      if (timer) {
          screen_output << string_format("%52s", shorter.c_str()) << " : ";
      } else {
          screen_output << string_format("%41s", shorter.c_str()) << " : ";
      }
#if defined(APEX_THROTTLE)
      if (!apex_options::use_tau()) {
        // if this profile was throttled, don't output the measurements.
        // they are limited and bogus, anyway.
        unordered_set<task_identifier>::const_iterator it4;
        {
              read_lock_type l(throttled_event_set_mutex);
            it4 = throttled_tasks.find(task_id);
          }
        if (it4!= throttled_tasks.end()) {
            screen_output << "DISABLED (high frequency, short duration)"
                << endl;
            return;
        }
      }
#endif
      if(p->get_calls() == 0 && p->get_times_reset() > 0) {
            screen_output << "Not called since reset." << endl;
            return;
      }
      if(p->get_calls() < 1 && p->get_times_reset() == 0) {
        p->get_profile()->calls = 1;
      }
      if (p->get_calls() < 999999) {
          screen_output << string_format(PAD_WITH_SPACES,
            to_string((int)p->get_calls()).c_str()) << "   " ;
      } else {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_calls())
            << "   " ;
      }
      if (include_stops) {
        auto stops = std::max<double>(0.0, (p->get_stops() - p->get_calls()));
        if (stops < 999999) {
            screen_output << string_format(PAD_WITH_SPACES,
                to_string((int)stops).c_str()) << "   " ;
        } else {
            screen_output << string_format(FORMAT_SCIENTIFIC, stops)
                << "   " ;
        }
      }
      if (p->get_type() == APEX_TIMER) {
        csv_output << "\"" << action_name << "\",";
        csv_output << llround(p->get_calls()) << ",";
        csv_output << std::llround(p->get_accumulated_useconds()) << ",";
        //screen_output << " --n/a--   " ;
        if (p->get_mean_seconds() > 10000) {
            screen_output << string_format(FORMAT_SCIENTIFIC,
                (p->get_mean_seconds())) << "   " ;
        } else {
            screen_output << string_format(FORMAT_PERCENT,
                (p->get_mean_seconds())) << "   " ;
        }
        //screen_output << " --n/a--   " ;
        if (p->get_accumulated_seconds() > 10000) {
            screen_output << string_format(FORMAT_SCIENTIFIC,
                (p->get_accumulated_seconds())) << "   " ;
        } else {
            screen_output << string_format(FORMAT_PERCENT,
                (p->get_accumulated_seconds())) << "   " ;
        }
        //screen_output << " --n/a--   " ;
        if (action_name.compare(APEX_MAIN_STR) == 0) {
            screen_output << string_format(FORMAT_PERCENT, 100.0);
        } else {
            total_accumulated += p->get_accumulated_seconds();
            double tmp = ((p->get_accumulated_seconds())
                /total_main)*100.0;
            if (tmp > 100.0) {
                screen_output << " --n/a--" ;
            } else {
                screen_output << string_format(FORMAT_PERCENT, tmp);
            }
        }
#if APEX_HAVE_PAPI
        if (include_papi) {
            for (int i = 0 ; i < num_papi_counters ; i++) {
                screen_output  << "   " << string_format(FORMAT_SCIENTIFIC,
                    (p->get_papi_metrics()[i]));
                csv_output << "," << std::llround(p->get_papi_metrics()[i]);
            }
        }
#endif
        if (apex_options::track_memory()) {
            if (p->get_allocations() > 999999) {
                screen_output  << "   " << string_format(FORMAT_SCIENTIFIC,
                    (p->get_allocations()));
            } else {
                screen_output  << "   " << string_format(PAD_WITH_SPACES,
                    to_string((int)p->get_allocations()).c_str());
            }
            csv_output << "," << std::llround(p->get_allocations());

            if (p->get_bytes_allocated() > 999999) {
                screen_output  << "   " << string_format(FORMAT_SCIENTIFIC,
                    (p->get_bytes_allocated()));
            } else {
                screen_output  << "   " << string_format(PAD_WITH_SPACES,
                    to_string((int)p->get_bytes_allocated()).c_str());
            }
            csv_output << "," << std::llround(p->get_bytes_allocated());

            if (p->get_frees() > 999999) {
                screen_output  << "   " << string_format(FORMAT_SCIENTIFIC,
                    (p->get_frees()));
            } else {
                screen_output  << "   " << string_format(PAD_WITH_SPACES,
                    to_string((int)p->get_frees()).c_str());
            }
            csv_output << "," << std::llround(p->get_frees());

            if (p->get_bytes_freed() > 999999) {
                screen_output  << "   " << string_format(FORMAT_SCIENTIFIC,
                    (p->get_bytes_freed()));
            } else {
                screen_output  << "   " << string_format(PAD_WITH_SPACES,
                    to_string((int)p->get_bytes_freed()).c_str());
            }
            csv_output << "," << std::llround(p->get_bytes_freed());
        }
        screen_output << endl;
        csv_output << endl;
      } else {
        csv_output << "\"" << action_name << "\",";
        csv_output << llround(p->get_calls()) << ",";
        csv_output << std::llround(p->get_minimum()) << ",";
        csv_output << std::llround(p->get_mean()) << ",";
        csv_output << std::llround(p->get_maximum()) << ",";
        csv_output << std::llround(p->get_stddev()) << endl;
        if (action_name.find('%') == string::npos && p->get_minimum() > 10000) {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_minimum()) << "   " ;
        } else {
          screen_output << string_format(FORMAT_PERCENT, p->get_minimum()) << "   " ;
        }
        if (action_name.find('%') == string::npos && p->get_mean() > 10000) {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_mean()) << "   " ;
        } else {
          screen_output << string_format(FORMAT_PERCENT, p->get_mean()) << "   " ;
        }
        if (action_name.find('%') == string::npos && p->get_maximum() > 10000) {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_maximum()) << "   " ;
        } else {
          screen_output << string_format(FORMAT_PERCENT, p->get_maximum()) << "   " ;
        }
        if (action_name.find('%') == string::npos && p->get_stddev() > 10000) {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_stddev()) << "   " ;
        } else {
          screen_output << string_format(FORMAT_PERCENT, p->get_stddev()) << "   " ;
        }
        screen_output << endl;
      }
  }

  bool timer_cmp(pair<std::string, apex_profile*>& a,
                 pair<std::string, apex_profile*>& b) {
      return a.second->accumulated > b.second->accumulated;
  }

  /* At program termination, write the measurements to the screen, or to CSV
   * file, or both. */
  void profiler_listener::finalize_profiles(dump_event_data &data, std::map<std::string, apex_profile*>& all_profiles) {
    if (apex_options::use_tau()) {
      tau_listener::Tau_start_wrapper("profiler_listener::finalize_profiles");
    }
    // our TOTAL available time is the elapsed * the number of threads, or cores
    int num_worker_threads = thread_instance::get_num_workers();
    auto main_id = task_identifier::get_main_task_id();
    profile * total_time = get_profile(*main_id);
#ifndef APEX_SYNCHRONOUS_PROCESSING
    /* The profiles haven't been processed yet. */
    while (total_time == nullptr) {
#ifdef APEX_HAVE_HPX
        // schedule an HPX action
        apex_schedule_process_profiles();
#else
        queue_signal.post();
#endif
        // wait for profiles to update
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        total_time = get_profile(main_id);
    }
#endif  // APEX_SYNCHRONOUS_PROCESSING
    double wall_clock_main = total_time->get_accumulated_seconds();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_hpx_threads = 0;
    double total_main = wall_clock_main *
        fmin(hardware_concurrency(), num_worker_threads);
    DEBUG_PRINT("%s:%d\n", __func__, __LINE__);
    // create a stringstream to hold all the screen output - we may not
    // want to write it out
    stringstream screen_output;
    // create a stringstream to hold all the CSV output - we may not
    // want to write it out
    stringstream csv_output;
    // iterate over the profiles in the address map
    screen_output << endl << "Elapsed time: " << wall_clock_main
        << " seconds" << endl;
    screen_output << "Total processes detected: " << apex::instance()->get_num_ranks()
        << endl;
    screen_output << "HW Threads detected on rank 0: " << hardware_concurrency()
        << endl;
    screen_output << "Worker Threads observed on rank 0: "
        << num_worker_threads << endl;
    screen_output << "Available CPU time on rank 0: "
        << total_main << " seconds" << endl;
    screen_output << "Available CPU time on all ranks: "
        << total_main * apex::instance()->get_num_ranks() << " seconds" << endl << endl;
    //double divisor = wall_clock_main; // could be total_main, for available CPU time.
    double divisor = total_main; // could be total_main, for available CPU time.

    double total_accumulated = 0.0;
    std::vector<std::string> id_vector;
    // iterate over the counters, and sort their names
    {
        std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
        for(auto it2 : all_profiles) {
            std::string name = it2.first;
            apex_profile * p = it2.second;
            if (p->type != APEX_TIMER) {
                id_vector.push_back(name);
            }
        }
    }
    csv_output << "\"counter\",\"num samples\",\"minimum\",\"mean\""
        << "\"maximum\",\"stddev\"" << endl;
    if (id_vector.size() > 0) {
        screen_output << "Counter                                   : "
        << "#samples | minimum |    mean  |  maximum |  stddev " << endl;
        //screen_output << "Counter                        : #samples | "
        //<< "minimum |    mean  |  maximum |   total  |  stddev " << endl;
        screen_output << "------------------------------------------"
        << "------------------------------------------------------" << endl;
        std::sort(id_vector.begin(), id_vector.end());
        // iterate over the counters
        for(auto name : id_vector) {
            auto p = all_profiles.find(name);
            if (p != all_profiles.end()) {
                profile tmp(p->second);
                write_one_timer(name, &tmp, screen_output, csv_output,
                    total_accumulated, divisor, false);
            }
        }
        screen_output << "------------------------------------------"
            << "------------------------------------------------------\n";
        screen_output << endl;
    }
    csv_output << "\n\n\"task\",\"num calls\",\"total microseconds\"";
#if APEX_HAVE_PAPI
    for (int i = 0 ; i < num_papi_counters ; i++) {
       csv_output << ",\"" << metric_names[i] << "\"";
    }
#endif
    if (apex_options::track_memory()) {
       csv_output << ",\"allocations\", \"bytes allocated\", \"frees\", \"bytes freed\"";
    }
    csv_output << endl;
    std::string re("PAPI_");
    std::string tmpstr(apex_options::papi_metrics());
    size_t index = 0;
    while (true) {
         /* Locate the substring to replace. */
         index = tmpstr.find(re, index);
         if (index == std::string::npos) break;
         /* Make the replacement. */
         tmpstr.replace(index, re.size(), "| ");
         /* Advance index forward so the next iteration doesn't pick it up as well. */
         index += 2;
    }

    // Declare vector of pairs
    vector<pair<std::string, apex_profile*> > timer_vector;
    // iterate over the timers
    for(auto& it2 : all_profiles) {
        apex_profile * p = it2.second;
        if (p->type == APEX_TIMER) {
            timer_vector.push_back(it2);
        }
    }
    // sort by accumulated value
    std::sort(timer_vector.begin(), timer_vector.end(), timer_cmp);

    screen_output << "GPU Timers                                           : "
        << "#calls  |    mean  |   total  |  % total  ";
    if (apex_options::track_memory()) {
       screen_output << "|  allocs |  (bytes) |    frees |   (bytes) ";
    }
    screen_output << endl;
    screen_output << "----------------------------------------------"
        << "--------------------------------------------------";

    screen_output << endl;
    // iterate over the timers
    for(auto& pair_itr : timer_vector) {
        std::string name = pair_itr.first;
        if (name.find("GPU: ", 0) == std::string::npos) continue;
        auto p = all_profiles.find(name);
        if (p != all_profiles.end()) {
            profile tmp(p->second);
            write_one_timer(name, &tmp, screen_output, csv_output,
                total_accumulated, divisor, true);
            if (name.compare(APEX_MAIN_STR) != 0) {
                total_hpx_threads = total_hpx_threads + tmp.get_calls();
            }
        }
    }

    screen_output << "--------------------------------------------------"
        << "----------------------------------------------";
    if (apex_options::track_memory()) {
        screen_output << "--------------------------------------------";
    }
    screen_output << endl;
    screen_output << endl;

    screen_output << "CPU Timers                                           : "
        << "#calls  |  #yields |    mean  |   total  |  % total  "
        << tmpstr;
    if (apex_options::track_memory()) {
       screen_output << "|  allocs |  (bytes) |    frees |   (bytes) ";
    }
    screen_output << endl;
    screen_output << "----------------------------------------------"
        << "-------------------------------------------------------------";
    if (apex_options::track_memory()) {
        screen_output << "--------------------------------------------";
    }
    screen_output << endl;

    // write the main timer
    std::string tmp_main(APEX_MAIN_STR);
    write_one_timer(tmp_main, total_time, screen_output, csv_output,
        total_accumulated, divisor, true, true, true);
    // iterate over the timers
    for(auto& pair_itr : timer_vector) {
        std::string name = pair_itr.first;
        if (name.find("GPU: ", 0) != std::string::npos) continue;
        auto p = all_profiles.find(name);
        if (p != all_profiles.end()) {
            profile tmp(p->second);
            write_one_timer(name, &tmp, screen_output, csv_output,
                total_accumulated, divisor, true, true, true);
            if (name.compare(APEX_MAIN_STR) != 0) {
                total_hpx_threads = total_hpx_threads + tmp.get_calls();
            }
        }
    }
    screen_output << "--------------------------------------------------"
        << "---------------------------------------------------------";
    if (apex_options::track_memory()) {
        screen_output << "--------------------------------------------";
    }
    screen_output << endl;
    screen_output << endl;

    double all_total_main = total_main * apex::instance()->get_num_ranks();
    double idle_rate = all_total_main - total_accumulated;
    if (idle_rate >= 0.0) {
      screen_output << string_format("%52s", APEX_IDLE_TIME) << " : ";
      // pad with spaces for #calls, mean
      screen_output << "                      ";
      if (idle_rate > 10000) {
        screen_output << string_format(FORMAT_SCIENTIFIC, idle_rate) << "   " ;
      } else {
        screen_output << string_format(FORMAT_PERCENT, idle_rate) << "   " ;
      }
      screen_output << string_format(FORMAT_PERCENT,
        ((idle_rate/all_total_main)*100)) << endl;
    }
    screen_output << "--------------------------------------------------"
        << "----------------------------------------------";
    if (apex_options::track_memory()) {
        screen_output << "--------------------------------------------";
    }
    screen_output << endl;

    screen_output << string_format("%52s", "Total timers") << " : ";
    std::stringstream total_ss;
    // the GCC 9 compiler on Apple doesn't do locale correctly!
#if !defined(__APPLE__) && (__GNUC__ != 9)
    total_ss.imbue(std::locale(""));
#endif
    total_ss << std::fixed << ((uint64_t)total_hpx_threads);
        screen_output << total_ss.str() << std::endl;

    if (apex_options::use_screen_output() && node_id == 0) {
        cout << screen_output.str();
        data.output = screen_output.str();
    }

    if (apex_options::use_csv_output()) {
        ofstream csvfile;
        stringstream csvname;
        csvname << apex_options::output_file_path();
        csvname << filesystem_separator() << "apex." << node_id << ".csv";
        // std::cout << "Writing: " << csvname.str() << std::endl;
        csvfile.open(csvname.str(), ios::out);
        csvfile << csv_output.str();
        csvfile.close();
    }
    if (apex_options::use_tau()) {
      tau_listener::Tau_stop_wrapper("profiler_listener::finalize_profiles");
    }
  }

  void profiler_listener::write_taskgraph(void) {
    std::cout << "Writing APEX taskgraph..." << std::endl;
    { // we need to lock in case another thread appears
        std::unique_lock<std::mutex> queue_lock(queue_mtx);
        // get all the remaining dependencies
        task_dependency* td;
        for (auto a_queue : dependency_queues) {
            while(a_queue->try_dequeue(td)) {
                process_dependency(td);
            }
        }
    }

    /* before calling parent.get_name(), make sure we create
     * a thread_instance object that is NOT a worker. */
    thread_instance::instance(false);
    ofstream myfile;
    stringstream dotname;
    dotname << apex_options::output_file_path();
    dotname << filesystem_separator() << "taskgraph." << node_id << ".dot";
    myfile.open(dotname.str().c_str());

    // our TOTAL available time is the elapsed * the number of threads, or cores
    int num_worker_threads = thread_instance::get_num_workers();
    auto main_id = task_identifier::get_main_task_id();
    profile * total_time = get_profile(*main_id);
    double wall_clock_main = total_time->get_accumulated_seconds();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_main = wall_clock_main * fmin(hardware_concurrency(),
        num_worker_threads);

    myfile << "digraph prof {\n";
    myfile << " label = \"Elapsed Time: " << wall_clock_main;
    myfile << " seconds\\lCores detected: " << hardware_concurrency();
    myfile << "\\lWorker threads observed: " << num_worker_threads;
    // is scaling this necessary?
    myfile << "\\lAvailable CPU time: " << total_main << " seconds\\l\"\n";
    myfile << " labelloc = \"t\";\n";
    myfile << " labeljust = \"l\";\n";
    myfile << " overlap = false;\n";
    myfile << " splines = true;\n";
    myfile << " rankdir = \"LR\";\n";
    myfile << " node [shape=box];\n";
    for(auto dep = task_dependencies.begin();
        dep != task_dependencies.end(); dep++) {
        task_identifier parent = dep->first;
        auto children = dep->second;
        string parent_name = parent.get_tree_name();
        for(auto offspring = children->begin();
            offspring != children->end(); offspring++) {
            task_identifier child = offspring->first;
            int count = offspring->second;
            string child_name = child.get_tree_name();
            myfile << "  \"" << parent_name << "\" -> \"" << child_name << "\"";
            myfile << " [ label=\"  count: " << count << "\" ]; " << std::endl;
        }
    }
    // delete the dependency graph
    for(auto dep = task_dependencies.begin();
        dep != task_dependencies.end(); dep++) {
        auto children = dep->second;
        children->clear();
        delete(children);
    }
    task_dependencies.clear();

    // output nodes with  "main" [shape=box; style=filled; fillcolor="#ff0000" ];
    unordered_map<task_identifier, profile*>::const_iterator it;
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it = task_map.begin(); it != task_map.end(); it++) {
      profile * p = it->second;
      // shouldn't happen, but?
      if (p == nullptr) continue;
      /*
      int divisor = num_worker_threads;
      std::string divided_label("time per thread: ");
      if (num_worker_threads > p->get_calls()) {
          divisor = p->get_calls();
          divided_label = "time per call: ";
      }
      */
      int divisor = p->get_calls();
      std::string divided_label("per call: ");
      if (p->get_type() == APEX_TIMER) {
        node_color * c = get_node_color_visible(
            p->get_accumulated_seconds(), 0.0, wall_clock_main);
            //p->get_accumulated_seconds()/divisor, 0.0, wall_clock_main);
        task_identifier task_id = it->first;
        double accumulated = p->get_accumulated_seconds();
        myfile << "  \"" << task_id.get_tree_name() <<
            "\" [shape=box; style=filled; fillcolor=\"#" <<
            setfill('0') << setw(2) << hex << c->convert(c->red) <<
            setfill('0') << setw(2) << hex << c->convert(c->green) <<
            setfill('0') << setw(2) << hex << c->convert(c->blue) <<
            "\"; label=\"" << task_id.get_tree_name() <<
            "\\l calls: " << p->get_calls() <<
            "\\l time: " << accumulated <<
            "s\\l " << divided_label << accumulated/divisor <<
            "s\\l\" ];" << std::endl;
        delete(c);
      }
    }
    myfile << "}\n";
    myfile.close();
  }

  /* When writing a TAU profile, get the appropriate TAU group */
  inline std::string get_TAU_group(task_identifier& task_id) {
    std::stringstream ss;
    ss << "GROUP=\"" << task_id.get_group() << "\" ";
    std::string group{ss.str()};
    return group;
  }

  /* When writing a TAU profile, write out a timer line */
  void format_line(ofstream &myfile, profile * p, task_identifier& task_id) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << ((p->get_accumulated_useconds())) << " ";
    myfile << ((p->get_accumulated_useconds())) << " ";
    myfile << 0 << " ";
    myfile << get_TAU_group(task_id);
    myfile << endl;
  }

  /* When writing a TAU profile, write out the main timer line */
  void format_line(ofstream &myfile, profile * p, double not_main) {
    double calls = p->get_calls() == 0 ? 1 : p->get_calls();
    myfile << calls << " ";
    myfile << 0 << " ";
    myfile << (std::max<double>(((p->get_accumulated_useconds())
        - not_main),0.0)) << " ";
    myfile << ((p->get_accumulated_useconds())) << " ";
    myfile << 0 << " ";
    myfile << "GROUP=\"TAU_USER\" ";
    myfile << endl;
  }

  /* When writing a TAU profile, write out a counter line */
  void format_counter_line(ofstream &myfile, profile * p) {
    myfile << p->get_calls() << " ";       // numevents
    myfile << p->get_maximum() << " ";     // max
    myfile << p->get_minimum() << " ";     // min
    myfile << p->get_mean() << " ";        // mean
    myfile << p->get_sum_squares() << " ";
    myfile << endl;
  }

  void profiler_listener::write_tasktree(void) {
    //std::cout << "Writing APEX tasktree..." << std::endl;
    /* before calling parent.get_name(), make sure we create
     * a thread_instance object that is NOT a worker. */
    thread_instance::instance(false);
    ofstream myfile;
    stringstream dotname;
    dotname << apex_options::output_file_path();
    dotname << filesystem_separator() << "tasktree." << node_id << ".dot";
    myfile.open(dotname.str().c_str());

    // our TOTAL available time is the elapsed * the number of threads, or cores
    int num_worker_threads = thread_instance::get_num_workers();
    auto main_id = task_identifier::get_main_task_id();
    profile * total_time = get_profile(*main_id);
    double wall_clock_main = total_time->get_accumulated_seconds();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_main = wall_clock_main * fmin(hardware_concurrency(),
        num_worker_threads);

    myfile << "digraph prof {\n";
    myfile << " label = \"Elapsed Time: " << wall_clock_main;
    myfile << " seconds\\lCores detected: " << hardware_concurrency();
    myfile << "\\lWorker threads observed: " << num_worker_threads;
    // is scaling this necessary?
    myfile << "\\lAvailable CPU time: " << total_main << " seconds\\l\";\n";
    myfile << " labelloc = \"t\";\n";
    myfile << " labeljust = \"l\";\n";
    myfile << " overlap = false;\n";
    myfile << " splines = true;\n";
    myfile << " rankdir = \"LR\";\n";
    myfile << " node [shape=box];\n";
    auto root = task_wrapper::get_apex_main_wrapper();
    // recursively write out the tree
    root->tree_node->writeNode(myfile, wall_clock_main);
    myfile << "}\n";
    myfile.close();
    // dump the tree to a human readable file
    stringstream txtname;
    txtname << apex_options::output_file_path();
    txtname << filesystem_separator() << "tasktree." << node_id << ".txt";
    myfile.open(txtname.str().c_str());
    root->tree_node->writeNodeASCII(myfile, wall_clock_main, 0);
    myfile.close();
  }

  /* Write TAU profiles from the collected data. */
  void profiler_listener::write_profile() {
    ofstream myfile;
    stringstream datname;
    // name format: profile.nodeid.contextid.threadid
    // We only write one profile per process
    datname << apex_options::output_file_path();
    datname << filesystem_separator() << "profile." << node_id << ".0.0";

    // name format: profile.nodeid.contextid.threadid
    myfile.open(datname.str().c_str());
    int counter_events = 0;

    // Determine number of counter events, as these need to be
    // excluded from the number of normal timers
    unordered_map<task_identifier, profile*>::const_iterator it2;
    {
        std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
        for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
            profile * p = it2->second;
            if(p->get_type() == APEX_COUNTER) {
                counter_events++;
            }
        }
    }
    size_t function_count = task_map.size() - counter_events;
    if (apex_options::use_tasktree_output()) {
        auto root = task_wrapper::get_apex_main_wrapper();
        function_count += (root->tree_node->getNodeCount() - 1);
    }

    // Print the normal timers to the profile file
    // 1504 templated_functions_MULTI_TIME
    myfile << function_count << " templated_functions_MULTI_TIME" << endl;
    // # Name Calls Subrs Excl Incl ProfileCalls #
    myfile << "# Name Calls Subrs Excl Incl ProfileCalls #" << endl;

    // Iterate over the profiles which are associated to a function
    // by name. Only output the regular timers now. Counters are
    // in a separate section, below.
    profile * mainp = nullptr;
    double not_main = 0.0;
    {
        std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
        for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
            profile * p = it2->second;
            task_identifier task_id = it2->first;
            if(p->get_type() == APEX_TIMER) {
                string action_name = task_id.get_name();
                if(action_name.compare(APEX_MAIN_STR) == 0) {
                    mainp = p;
                } else {
                    myfile << "\"" << action_name << "\" ";
                    format_line (myfile, p, task_id);
                    not_main += (p->get_accumulated_useconds());
                }
            }
        }
        if (mainp != nullptr) {
            myfile << "\".TAU application\" ";
            format_line (myfile, mainp, not_main);
        }
    }

    // If we maintained the tasktree, we can write out the callpath.
    if (apex_options::use_tasktree_output()) {
        auto root = task_wrapper::get_apex_main_wrapper();
        std::string prefix{""};
        root->tree_node->writeTAUCallpath(myfile, prefix);
    }

    // 0 aggregates
    myfile << "0 aggregates" << endl;

    // Now process the counters, if there are any.
    if(counter_events > 0) {
      myfile << counter_events << " userevents" << endl;
      myfile << "# eventname numevents max min mean sumsqr" << endl;
      for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
        profile * p = it2->second;
        if(p->get_type() == APEX_COUNTER) {
          task_identifier task_id = it2->first;
          myfile << "\"" << task_id.get_name() << "\" ";
          format_counter_line (myfile, p);
        }
      }
    }
    myfile.close();
  }

  /*
   * The main function for the consumer thread has to be static, but
   * the processing needs access to member variables, so get the
   * profiler_listener instance, and call it's proper function.
   *
   * This is a wrapper, so that we can launch the thread and set
   * affinity. However, process_profiles_wrapper is also used by the
   * last worker that calls apex_finalize(), so we don't want to change
   * that thread's affinity. So this wrapper is only for the consumer
   * thread.
   */
  void profiler_listener::consumer_process_profiles_wrapper(void) {
      if (apex_options::pin_apex_threads()) {
            set_thread_affinity();
      }
      process_profiles_wrapper();
  }

  /*
   * The main function for the consumer thread has to be static, but
   * the processing needs access to member variables, so get the
   * profiler_listener instance, and call it's proper function.
   */
  void profiler_listener::process_profiles_wrapper(void) {
      apex * inst = apex::instance();
      if (inst != nullptr) {
          profiler_listener * pl = inst->the_profiler_listener;
          if (pl != nullptr) {
#ifdef APEX_TRACE_APEX
              scoped_timer p("apex::process_profiles");
#endif
              pl->process_profiles();
          }
      }
      consumer_task_running.clear(memory_order_release);
  }

  bool profiler_listener::concurrent_cleanup(int i){
      //set_thread_affinity(i);
      std::shared_ptr<profiler> p;
      while(allqueues[i]->try_dequeue(p)) {
             process_profile(p,0);
      }
      return true;
  }

  /* This is the main function for the consumer thread.
   * Operation outside of HPX:
   * It will wait at a semaphore for pending work. When there is
   * work on one or more queues, it will iterate over the queues
   * and process the pending profiler objects, updating the profiles
   * as it goes.
   *
   * Operation inside of HPX:
   * This function gets called as an HPX task when there is new
   * work to be processed.  It will exit after processing
   * 1000 profiler timers, so as not to stay in this function
   * too long, occupying an HPX thread.  If it terminates before
   * clearing the queue, it will schedule a new HPX task.
   *
   * */
  void profiler_listener::process_profiles(void)
  {
    if (!_initialized) {
      initialize_worker_thread_for_tau();
      _initialized = true;
    }
    if (apex_options::use_tau()) {
      tau_listener::Tau_start_wrapper("profiler_listener::process_profiles");
    }
    /*
    static auto prof = new_task(__func__);
    start(prof);
    */

    std::shared_ptr<profiler> p;
    task_dependency* td;
#ifdef APEX_HAVE_HPX
    //bool schedule_another_task = false;
    {
        size_t num_queues = 0;
        {
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            num_queues = allqueues.size();
        }
        for (size_t q = 0 ; q < num_queues ; q++) {
            int i = 0;
            while(!_done && allqueues[q]->try_dequeue(p)) {
                process_profile(p, 0);
                /*
                if (++i > 1000 && !synchronous_flush) {
                    schedule_another_task = true;
                    break;
                }
                */
            }
        }
    }
    if (apex_options::use_taskgraph_output()) {
        size_t num_queues = 0;
        {
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            num_queues = dependency_queues.size();
        }
        for (size_t q = 0 ; q < num_queues ; q++) {
            int i = 0;
            while(!_done && dependency_queues[q]->try_dequeue(td)) {
                process_dependency(td);
                /*
                if (++i > 1000 && !synchronous_flush) {
                    schedule_another_task = true;
                    break;
                }
                */
            }
        }
    }
#else
    // Main loop. Stay in this loop unless "done".
    while (!_done) {
        queue_signal.wait();
        if (apex_options::use_tau()) {
            tau_listener::Tau_start_wrapper(
                "profiler_listener::process_profiles: main loop");
        }
        {
            size_t num_queues = 0;
                std::unique_lock<std::mutex> queue_lock(queue_mtx);
            {
                num_queues = allqueues.size();
            }
            for (size_t q = 0 ; q < num_queues ; q++) {
                while(!_done && allqueues[q]->try_dequeue(p)) {
                    process_profile(p, 0);
                }
            }
        }
        if (apex_options::use_taskgraph_output()) {
            size_t num_queues = 0;
                std::unique_lock<std::mutex> queue_lock(queue_mtx);
            {
                num_queues = dependency_queues.size();
            }
            for (size_t q = 0 ; q < num_queues ; q++) {
                while(!_done && dependency_queues[q]->try_dequeue(td)) {
                    process_dependency(td);
                }
            }
        }
        if (apex_options::use_tau()) {
            tau_listener::Tau_stop_wrapper(
                "profiler_listener::process_profiles: main loop");
        }
        // release the flag, and wait for the queue_signal
        consumer_task_running.clear(memory_order_release);
    }
#endif

#ifdef APEX_HAVE_HPX // don't hang out in this task too long.
    /*
    if (schedule_another_task) {
        apex_schedule_process_profiles();
    }
    */
#endif

    //stop(prof);
    if (apex_options::use_tau()) {
      tau_listener::Tau_stop_wrapper("profiler_listener::process_profiles");
    }
  }

#if APEX_HAVE_PAPI
#define PAPI_ERROR_CHECK(name) \
if (rc != 0) cout << "PAPI error! " << name << ": " << PAPI_strerror(rc) << endl;

// THIS MACRO EXITS if the papi call does not return PAPI_OK. Do not use for routines that
// return anything else; e.g. PAPI_num_components, PAPI_get_component_info, PAPI_library_init.
#define CALL_PAPI_OK(papi_routine)                                                        \
    do {                                                                                  \
        int _papiret = papi_routine;                                                      \
        if (_papiret != PAPI_OK) {                                                        \
            fprintf(stderr, "%s:%d macro: PAPI Error: function " #papi_routine " failed with ret=%d [%s].\n", \
                    __FILE__, __LINE__, _papiret, PAPI_strerror(_papiret));               \
            abort();                                                                     \
        }                                                                                 \
    } while (0);

  void profiler_listener::initialize_PAPI(bool first_time) {
      /* Do we have any metrics?  If not, return */
      if (strlen(apex_options::papi_metrics()) == 0) {
        return;
      }

      /* Initialize PAPI */
      if (first_time) {
        int ver = PAPI_library_init( PAPI_VER_CURRENT );
        if (ver != PAPI_VER_CURRENT) {
            fprintf(stderr, "PAPI_library_init() failed with version=%d, expected %d.\n",
                    ver, PAPI_VER_CURRENT);
            exit(-1);
        }
        //rc = PAPI_multiplex_init(); // use more counters than allowed
        //PAPI_ERROR_CHECK("PAPI_multiplex_init");
        CALL_PAPI_OK(PAPI_thread_init( &thread_instance::get_id ));
        // default
        //rc = PAPI_set_domain(PAPI_DOM_ALL);
        //PAPI_ERROR_CHECK("PAPI_set_domain");
      } else {
        CALL_PAPI_OK(PAPI_register_thread());
      }

      /* First, we need to tokenize the list of metrics */
      std::stringstream tmpstr(apex_options::papi_metrics());
      // use stream iterators to copy the stream to the vector as whitespace
      // separated strings
      std::istream_iterator<std::string> tmpstr_it(tmpstr);
      std::istream_iterator<std::string> tmpstr_end;
      std::vector<std::string> tmpstr_results(tmpstr_it, tmpstr_end);

      /* Second, we need to split the metrics into component sets */
      std::map<std::string, std::vector<std::string> > component_maps;
      // iterate over the counter names in the vector
      for (auto p : tmpstr_results) {
        // is this a PAPI preset?
        std::string papi_s{"PAPI"};
        if (p.rfind("PAPI_", 0) == 0) {
            //std::cout << "Found PAPI component metric" << std::endl;
            if (component_maps.count(papi_s) == 0) {
                std::vector<std::string> tmp_metrics;
                //std::cout << "component metric: " << p << std::endl;
                tmp_metrics.push_back(p);
                component_maps.insert(
                    std::make_pair(papi_s, tmp_metrics));
            } else {
                component_maps[papi_s].push_back(p);
            }
        } else {
            //std::cout << "Found other component metric" << std::endl;
            //std::cout << "component metric: " << p << std::endl;
            /* not a PAPI preset, so this is a component metric */
            auto index = p.find(":::");
            if (index != std::string::npos) {
                /* get the component name */
                std::string c_name = p;
                c_name.erase(index);
                //std::cout << "component : " << c_name << std::endl;
                if (component_maps.count(c_name) == 0) {
                    std::vector<std::string> tmp_metrics;
                    tmp_metrics.push_back(p);
                    component_maps.insert(
                        std::make_pair(c_name, tmp_metrics));
                } else {
                    component_maps[c_name].push_back(p);
                }
            }
        }
      }
      /* For each component set, create an event set and add the metric */
      for (auto c : component_maps) {
        std::string name = c.first;
        auto metrics = c.second;

        int EventSet = PAPI_NULL;
      CALL_PAPI_OK(PAPI_create_eventset(&EventSet));
      // default
      //rc = PAPI_assign_eventset_component (EventSet, 0);
      //PAPI_ERROR_CHECK("PAPI_assign_eventset_component");
      // default
      //rc = PAPI_set_granularity(PAPI_GRN_THR);
      //PAPI_ERROR_CHECK("PAPI_set_granularity");
      // unnecessary complexity
      //rc = PAPI_set_multiplex(EventSet);
      //PAPI_ERROR_CHECK("PAPI_set_multiplex");
      // parse the requested set of papi counters
      // The string is modified by strtok, so copy it.
        int code;
        // iterate over the counter names in the vector
        for (auto p : metrics) {
          CALL_PAPI_OK(PAPI_event_name_to_code(const_cast<char*>(p.c_str()), &code));
          if (PAPI_query_event (code) == PAPI_OK) {
            CALL_PAPI_OK(PAPI_add_event(EventSet, code));
            if (first_time) {
              metric_names.push_back(string(p.c_str()));
              num_papi_counters++;
            }
          }
        }
        if (!apex_options::papi_suspend()) {
            CALL_PAPI_OK(PAPI_start( EventSet ));
            _pls.thread_papi_state = papi_running;
        }
        _pls.event_sets.push_back(EventSet);
        _pls.event_set_sizes.push_back(metrics.size());
      }
  }

#endif

  /* When APEX gets a STARTUP event, do some initialization. */
  void profiler_listener::on_startup(startup_event_data &data) {
    if (!_done) {
      _pls.my_tid = (unsigned int)thread_instance::get_id();
      async_thread_setup();
#ifndef APEX_SYNCHRONOUS_PROCESSING
#ifndef APEX_HAVE_HPX
      // Start the consumer thread, to process profiler objects.
      consumer_thread = new std::thread(consumer_process_profiles_wrapper);
#endif
#endif // APEX_SYNCHRONOUS_PROCESSING

#if APEX_HAVE_PAPI
      initialize_PAPI(true);
#endif

      /* This commented out code is to change the priority of the consumer thread.
       * IDEALLY, I would like to make this a low priority thread, but that is as
       * yet unsuccessful. */
#if 0
      int retcode;
      int policy;

      pthread_t threadID = (pthread_t) consumer_thread->native_handle();

      struct sched_param param;

      if ((retcode = pthread_getschedparam(threadID, &policy, &param)) != 0)
      {
        errno = retcode;
        perror("pthread_getschedparam");
        exit(EXIT_FAILURE);
      }
      std::cout << "INHERITED: ";
      std::cout << "policy=" << ((policy == SCHED_FIFO)  ? "SCHED_FIFO" :
          (policy == SCHED_RR)    ? "SCHED_RR" :
          (policy == SCHED_OTHER) ? "SCHED_OTHER" :
          "???")
        << ", priority=" << param.sched_priority << " of " <<
        sched_get_priority_min(policy) << "," << sched_get_priority_max(policy)
        << std::endl;
      //param.sched_priority = 10;
      if ((retcode = pthread_setschedparam(threadID, policy, &param)) != 0)
      {
        errno = retcode;
        perror("pthread_setschedparam");
        exit(EXIT_FAILURE);
      }
#endif

      // time the whole application.
      main_timer = std::make_shared<profiler>(
        task_wrapper::get_apex_main_wrapper());
#if APEX_HAVE_PAPI
      if (num_papi_counters > 0 && !apex_options::papi_suspend() &&
        _pls.thread_papi_state == papi_running) {
        size_t index = 0;
        for (size_t i = 0 ; i < _pls.event_sets.size() ; i++) {
            CALL_PAPI_OK(PAPI_read( _pls.event_sets[i], &(main_timer->papi_start_values[index]) ));
            index = index + _pls.event_set_sizes[i];
        }
      }
#endif
    }
    node_id = data.comm_rank;
  }

  /* On the dump event, output all the profiles regardless of whether
   * the screen dump flag is set. */
  void profiler_listener::on_dump(dump_event_data &data) {
    if (_done) { return; }

    if (!_main_timer_stopped) {
        // stop the main timer, and process that profile?
        yield_main_timer();
        push_profiler((unsigned int)thread_instance::get_id(), *main_timer);
        // restart the main timer
        resume_main_timer();
    }

    // trigger statistics updating
#ifndef APEX_SYNCHRONOUS_PROCESSING
#ifdef APEX_HAVE_HPX
    // We can't schedule an action, because the runtime might be gone
    // if we are in the dump() during finalize.  So synchronously
    // process the queue.
    // synchronous_flush = true;
    process_profiles_wrapper();
    // synchronous_flush = false;
#else
    queue_signal.post();
    // wait until any other threads are done processing dependencies
    while(consumer_task_running.test_and_set(memory_order_acq_rel)) { }
#endif
#endif // APEX_SYNCHRONOUS_PROCESSING

      // output to screen?
      if (apex_options::use_screen_output() ||
          apex_options::use_taskgraph_output() ||
          apex_options::use_tasktree_output() ||
          apex_options::use_csv_output())
      {
        size_t ignored = 0;
        { // we need to lock in case another thread appears
            size_t num_queues = 0;
            {
                std::unique_lock<std::mutex> queue_lock(queue_mtx);
                num_queues = allqueues.size();
            }
            for (size_t q = 0 ; q < num_queues ; q++) {
                ignored += allqueues[q]->size_approx();
            }
        }
        if (ignored > 100000) {
          std::cout << "Info: " << ignored
            << " items remaining on on the profiler_listener queue...";
            fflush(stderr);
        }
        /* APEX can't handle spawning a bunch of new APEX threads at this time,
         * so just process the queue. Anyway, it shouldn't get backed up that
         * much without suggesting there is a bigger problem. */
        {
            size_t num_queues = 0;
            {
                std::unique_lock<std::mutex> queue_lock(queue_mtx);
                num_queues = allqueues.size();
            }
            for (unsigned int i=0 ; i < num_queues ; ++i) {
                if (apex_options::use_tau()) {
                    tau_listener::Tau_start_wrapper(
                        "profiler_listener::concurrent_cleanup");
                }
                concurrent_cleanup(i);
                if (apex_options::use_tau()) {
                    tau_listener::Tau_stop_wrapper(
                        "profiler_listener::concurrent_cleanup");
                }
            }
        }
        if (ignored > 100000) {
          std::cerr << "done." << std::endl;
        }
      }
      if (apex_options::use_screen_output()) {
        DEBUG_PRINT("%s:%d:%d\n", __func__, __LINE__, node_id);
        // reduce/gather all profiles from all ranks
        auto reduced = reduce_profiles();
        DEBUG_PRINT("%s:%d:%d\n", __func__, __LINE__, node_id);
        if (node_id == 0) {
          if (apex_options::process_async_state()) {
            finalize_profiles(data, reduced);
          }
        }
      }
      if (apex_options::use_taskgraph_output())
      {
        write_taskgraph();
      }
      else if (apex_options::use_tasktree_output())
      {
        write_tasktree();
      }

      // output to 1 TAU profile per process?
      if (apex_options::use_profile_output() && !apex_options::use_tau()) {
        write_profile();
      }
      if (apex_options::task_scatterplot()) {
          task_scatterplot_sample_file() << task_scatterplot_samples.rdbuf();
          task_scatterplot_sample_file().close();
          counter_scatterplot_sample_file() << counter_scatterplot_samples.rdbuf();
          counter_scatterplot_sample_file().close();
      }
      if (data.reset) {
          reset_all();
      }
#ifndef APEX_SYNCHRONOUS_PROCESSING
      // on_dump() releasing the "task_running" flag
      consumer_task_running.clear(memory_order_release);
#endif
  }

  void profiler_listener::on_reset(task_identifier * id) {
    if (id == nullptr) {
        reset_all();
    } else {
        reset(id);
    }
  }

  /* On the shutdown event, notify the consumer thread that we are done
   * and set the "terminate" flag. */
  void profiler_listener::on_shutdown(shutdown_event_data &data) {
    APEX_UNUSED(data);
    if (_done) { return; }
    if (!_done) {
      _done = true;
      //node_id = data.node_id;
      //sleep(1);
#ifndef APEX_SYNCHRONOUS_PROCESSING
#ifndef APEX_HAVE_HPX
      queue_signal.post();
      queue_signal.dump_stats();
      if (consumer_thread != nullptr) {
          queue_signal.post(); // one more time, just to be sure
          consumer_thread->join();
      }
#endif
#endif // APEX_SYNCHRONOUS_PROCESSING

    }
  }

  /* When a new node is created */
  void profiler_listener::on_new_node(node_event_data &data) {
    if (!_done) {
    }
    APEX_UNUSED(data);
  }

  /* When a new thread is registered, expand all of our storage as necessary
   * to handle the new thread */
  void profiler_listener::on_new_thread(new_thread_event_data &data) {
    if (!_done) {
      _pls.my_tid = (unsigned int)thread_instance::get_id();
      async_thread_setup();
#if APEX_HAVE_PAPI
      initialize_PAPI(false);
#endif
    }
    APEX_UNUSED(data);
  }

  //extern "C" int main (int, char**);

  /* When a start event happens, create a profiler object. Unless this
   * named event is throttled, in which case do nothing, as quickly as possible */
  inline bool profiler_listener::_common_start(std::shared_ptr<task_wrapper>
    &tt_ptr, bool is_resume) {
    //std::cout << "Starting " << tt_ptr->get_task_id()->get_name() << std::endl;
    if (!_done) {
#if defined(APEX_THROTTLE)
      if (!apex_options::use_tau()) {
        // if this timer is throttled, return without doing anything
        unordered_set<task_identifier>::const_iterator it;
        {
              read_lock_type l(throttled_event_set_mutex);
            it = throttled_tasks.find(*tt_ptr->get_task_id());
        }
        if (it != throttled_tasks.end()) {
            /*
            * The throw is removed, because it is a performance penalty on some
            * systems on_start now returns a boolean
            */
            // to be caught by apex::start/resume
            //throw disabled_profiler_exception();
            return false;
        }
      }
#endif
      // start the profiler object, which starts our timers
      //std::shared_ptr<profiler> p = std::make_shared<profiler>(tt_ptr,
      //is_resume);
      // get the right task identifier, based on whether there are aliases
      profiler * p = new profiler(tt_ptr, is_resume);
      p->guid = tt_ptr->guid;
      thread_instance::instance().set_current_profiler(p);
#if APEX_HAVE_PAPI
      if (num_papi_counters > 0 && !apex_options::papi_suspend()) {
          // if papi was previously suspended, we need to start the counters
          if (_pls.thread_papi_state == papi_suspended) {
            for (size_t i = 0 ; i < _pls.event_sets.size() ; i++) {
                CALL_PAPI_OK(PAPI_start( _pls.event_sets[i] ));
            }
            _pls.thread_papi_state = papi_running;
          }
          size_t index = 0;
          for (size_t i = 0 ; i < _pls.event_sets.size() ; i++) {
              CALL_PAPI_OK(PAPI_read( _pls.event_sets[i], &(p->papi_start_values[index]) ));
              index = index + _pls.event_set_sizes[i];
          }
      } else {
          // if papi is still running, stop the counters
          if (_pls.thread_papi_state == papi_running) {
            long long dummy[8];
            for (size_t i = 0 ; i < _pls.event_sets.size() ; i++) {
                CALL_PAPI_OK(PAPI_stop( _pls.event_sets[i], dummy ));
            }
            _pls.thread_papi_state = papi_suspended;
          }
      }
#endif
    } else {
        return false;
    }
    return true;
  }

  inline void profiler_listener::push_profiler(int my_tid, profiler& p) {
      APEX_UNUSED(my_tid);
      // if we aren't processing profiler objects, just return.
      if (!apex_options::process_async_state()) { return; }
#ifdef APEX_TRACE_APEX
      if (p.get_task_id()->name == "apex::process_profiles_sync") { return; }
#endif
      process_profile(p,0);
      return;
  }

  inline void profiler_listener::push_profiler(int my_tid,
    std::shared_ptr<profiler> &p) {
      APEX_UNUSED(my_tid);
      // if we aren't processing profiler objects, just return.
      if (!apex_options::process_async_state()) { return; }
#ifdef APEX_TRACE_APEX
      if (p->get_task_id()->name == "apex::process_profiles_async") { return; }
#endif
      thequeue()->enqueue(p);
#ifndef APEX_HAVE_HPX
      // Check to see if the consumer is already running, to avoid calling
      // "post" too frequently - it is rather costly.
      if(!consumer_task_running.test_and_set(memory_order_acq_rel)) {
        queue_signal.post();
      }
#else
      // only fire off an action 0.1% of the time.
      static int thresh = RAND_MAX/1000;
      if (std::rand() < thresh) {
        apex_schedule_process_profiles();
      }
#endif
  }

  /* Stop the timer, if applicable, and queue the profiler object */
  inline void profiler_listener::_common_stop(std::shared_ptr<profiler> &p,
    bool is_yield) {
    if (!_done) {
      if (p) {
        p->stop(is_yield);
#if APEX_HAVE_PAPI
        if (num_papi_counters > 0 && !apex_options::papi_suspend() &&
            _pls.thread_papi_state == papi_running) {
            size_t index = 0;
            for (size_t i = 0 ; i < _pls.event_sets.size() ; i++) {
                CALL_PAPI_OK(PAPI_read( _pls.event_sets[i], &(p->papi_stop_values[index]) ));
                index = index + _pls.event_set_sizes[i];
            }
        }
#endif
#ifdef APEX_SYNCHRONOUS_PROCESSING
        push_profiler(_pls.my_tid, *p);
#else // APEX_SYNCHRONOUS_PROCESSING
        push_profiler(_pls.my_tid, p);
#endif // APEX_SYNCHRONOUS_PROCESSING
      }
    }
  }

  /* Start the timer */
  bool profiler_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    return _common_start(tt_ptr, false);
  }

  /* This is just like starting a timer, but don't increment the number of calls
   * value. That is because we are restarting an existing timer. */
  bool profiler_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
    return _common_start(tt_ptr, true);
  }

   /* Stop the timer */
  void profiler_listener::on_stop(std::shared_ptr<profiler> &p) {
    _common_stop(p, p->is_resume); // don't change the yield/resume value!
  }

  /* Stop the timer, but don't increment the number of calls */
  void profiler_listener::on_yield(std::shared_ptr<profiler> &p) {
    _common_stop(p, true);
  }

  /* When a thread exits, pop and stop all timers. */
  void profiler_listener::on_exit_thread(event_data &data) {
    APEX_UNUSED(data);
  }

  /* When an asynchronous thread is launched, they should
   * call apex::async_thread_setup() which will end up here.*/
  void profiler_listener::async_thread_setup(void) {
      // for asynchronous threads, check to make sure there is a queue!
      thequeue();
      if (apex_options::use_taskgraph_output()) {
        dependency_queue();
      }
  }

  /* When a sample value is processed, save it as a profiler object, and queue it. */
  void profiler_listener::on_sample_value(sample_value_event_data &data) {
    if (!_done) {
      // don't make a shared pointer if not necessary!
#ifdef APEX_SYNCHRONOUS_PROCESSING
      profiler p(task_identifier::get_task_id(
        *data.counter_name), data.counter_value);
      p.is_counter = data.is_counter;
#else // APEX_SYNCHRONOUS_PROCESSING
      std::shared_ptr<profiler> p =
        std::make_shared<profiler>(task_identifier::get_task_id(
        *data.counter_name), data.counter_value);
      p->is_counter = data.is_counter;
#endif // APEX_SYNCHRONOUS_PROCESSING
      push_profiler(_pls.my_tid, p);
    }
  }

  void profiler_listener::on_task_complete(std::shared_ptr<task_wrapper>
    &tt_ptr) {
    //printf("New task: %llu\n", task_id); fflush(stdout);
    if (!apex_options::use_taskgraph_output()) { return; }
    // get the right task identifier, based on whether there are aliases
    task_identifier * id = tt_ptr->get_task_id();
    // if the parent task is not null, use it (obviously)
    if (tt_ptr->parent != nullptr) {
        task_identifier * pid = tt_ptr->parent->get_task_id();
        dependency_queue()->enqueue(new task_dependency(pid, id));
        return;
    }
  }

  /* Communication send event. Save the number of bytes. */
  void profiler_listener::on_send(message_event_data &data) {
    if (!_done) {
      // don't make a shared pointer if not necessary!
#ifdef APEX_SYNCHRONOUS_PROCESSING
      profiler p(task_identifier::get_task_id("Bytes Sent"), (double)data.size);
#else // APEX_SYNCHRONOUS_PROCESSING
      std::shared_ptr<profiler> p = std::make_shared<profiler>(
        task_identifier::get_task_id("Bytes Sent"), (double)data.size);
#endif // APEX_SYNCHRONOUS_PROCESSING
      push_profiler(0, p);
    }
  }

  /* Communication recv event. Save the number of bytes. */
  void profiler_listener::on_recv(message_event_data &data) {
    if (!_done) {
      // don't make a shared pointer if not necessary!
#ifdef APEX_SYNCHRONOUS_PROCESSING
      profiler p(task_identifier::get_task_id("Bytes Received"), (double)data.size);
#else // APEX_SYNCHRONOUS_PROCESSING
      std::shared_ptr<profiler> p = std::make_shared<profiler>(
        task_identifier::get_task_id("Bytes Received"), (double)data.size);
#endif // APEX_SYNCHRONOUS_PROCESSING
      push_profiler(0, p);
    }
  }

  /* For periodic stuff. Do something? */
  void profiler_listener::on_periodic(periodic_event_data &data) {
    if (!_done) {
    }
    APEX_UNUSED(data);
  }

  /* For custom event stuff. Do something? */
  void profiler_listener::on_custom_event(custom_event_data &data) {
    if (!_done) {
    }
    APEX_UNUSED(data);
  }

  void profiler_listener::reset(task_identifier * id) {
    // don't make a shared pointer if not necessary!
#ifdef APEX_SYNCHRONOUS_PROCESSING
    profiler p(id, false, reset_type::CURRENT);
#else // APEX_SYNCHRONOUS_PROCESSING
    std::shared_ptr<profiler> p =
        std::make_shared<profiler>(id, false, reset_type::CURRENT);
#endif // APEX_SYNCHRONOUS_PROCESSING
    push_profiler(_pls.my_tid, p);
  }

  profiler_listener::~profiler_listener (void) {
      _done = true; // yikes!
      finalize();
      delete_profiles();
#ifndef APEX_SYNCHRONOUS_PROCESSING
#ifndef APEX_HAVE_HPX
#ifndef APEX_STATIC // unbelievable.  Deleting this object can crash in a static link.
      delete consumer_thread;
#endif
#endif
#endif // APEX_SYNCHRONOUS_PROCESSING
    std::unique_lock<std::mutex> queue_lock(queue_mtx);
    while (allqueues.size() > 0) {
        auto tmp = allqueues.back();
        allqueues.pop_back();
        delete(tmp);
    }
    while (dependency_queues.size() > 0) {
        auto tmp = dependency_queues.back();
        dependency_queues.pop_back();
        delete(tmp);
    }
    for (auto tmp : free_profiles) {
        delete(tmp);
    }
  }

  void profiler_listener::yield_main_timer(void) {
    if (!_main_timer_stopped) {
      APEX_ASSERT(main_timer != nullptr);
      main_timer->stop(true);
    }
  }

  void profiler_listener::resume_main_timer(void) {
      APEX_ASSERT(main_timer != nullptr);
      main_timer->restart();
  }

  void profiler_listener::increment_main_timer_allocations(double bytes) {
      APEX_ASSERT(main_timer != nullptr);
      main_timer->allocations++;
      main_timer->bytes_allocated += bytes;
  }

  void profiler_listener::increment_main_timer_frees(double bytes) {
      APEX_ASSERT(main_timer != nullptr);
      main_timer->frees++;
      main_timer->bytes_freed += bytes;
  }

  void profiler_listener::stop_main_timer(void) {
    if (!_main_timer_stopped) {
        APEX_ASSERT(main_timer != nullptr);
        main_timer->stop(true);
        //_common_stop(main_timer, false);
        //on_task_complete(main_timer->tt_ptr);
        _main_timer_stopped = true;
    }
  }

  void profiler_listener::on_pre_shutdown(void) {
    stop_main_timer();
  }

  void profiler_listener::push_profiler_public(std::shared_ptr<profiler> &p) {
    in_apex prevent_deadlocks;
    push_profiler(0, p);
  }

}

#ifdef APEX_HAVE_HPX
void apex_schedule_process_profiles() {
    if(get_hpx_runtime_ptr() == nullptr) return;
    if(!thread_instance::is_worker()) return;
    if(hpx_shutdown) {
        APEX_TOP_LEVEL_PACKAGE::profiler_listener::process_profiles_wrapper();
    } else {
        if(!consumer_task_running.test_and_set(memory_order_acq_rel)) {
            try {
                hpx::apply(
                    hpx::util::annotated_function(
                        &profiler_listener::process_profiles_wrapper,
                        "apex::profiler_listener::process_profiles"));
            } catch(...) {
                // During shutdown, we can't schedule a new task,
                // so we process profiles ourselves.
                profiler_listener::process_profiles_wrapper();
            }
        }
    }
}

#endif


