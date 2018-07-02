//  Copyright (c) 2014 University of Oregon
//

#ifdef APEX_HAVE_HPX
#include <hpx/config.hpp>
#ifdef APEX_HAVE_OTF2
#define APEX_TRACE_APEX
#endif // APEX_HAVE_OTF2
#endif // APEX_HAVE_HPX

#include "profiler_listener.hpp"
#include "profiler.hpp"
#include "task_wrapper.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include "apex_options.hpp"
#include "profile.hpp"
#include "apex.hpp"

#include <atomic>
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#include <unistd.h>
#include <sched.h>
#endif
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
#ifdef APEX_USE_CLOCK_TIMESTAMP
#define APEX_THROTTLE_PERCALL 0.00001 // 10 microseconds.
#else
#define APEX_THROTTLE_PERCALL 50000 // 50k cycles.
#endif
#endif

#if APEX_HAVE_PAPI
#include "papi.h"
#include <mutex>
std::mutex event_set_mutex;
#endif

#ifdef APEX_HAVE_HPX
#include <boost/assign.hpp>
#include <boost/cstdint.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos/local/composable_guard.hpp>
static void apex_schedule_process_profiles(void); // not in apex namespace
const int num_non_worker_threads_registered = 1; // including the main thread
#endif

#define APEX_MAIN "APEX MAIN"

#include "tau_listener.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <ctime>

#include <iomanip>
#include <locale>

using namespace std;
using namespace apex;

APEX_NATIVE_TLS unsigned int my_tid = 0; // the current thread's TID in APEX

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
        static APEX_NATIVE_TLS dependency_queue_t * _thequeue = _construct_dependency_queue();
        return _thequeue;
    }

  /* THis is a special profiler, indicating that the timer requested is
     throttled, and shouldn't be processed. */
  profiler* profiler::disabled_profiler = new profiler();

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
    return non_idle_time*profiler::get_cpu_mhz();
  }

  profile * profiler_listener::get_idle_time() {
    double non_idle_time = get_non_idle_time();
    /* Subtract the accumulated time from the main time span. */
    int num_worker_threads = thread_instance::get_num_threads();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>
           (MYCLOCK::now() - main_timer->start);
    double total_main = time_span.count() *
                fmin(hardware_concurrency(), num_worker_threads);
    double elapsed = total_main - non_idle_time;
    elapsed = elapsed > 0.0 ? elapsed : 0.0;
    profile * theprofile = new profile(elapsed*profiler::get_cpu_mhz(), 0, NULL, false);
    {
        std::unique_lock<std::mutex> l(free_profile_set_mutex);
        free_profiles.insert(theprofile);
    }
    return theprofile;
  }

  profile * profiler_listener::get_idle_rate() {
    double non_idle_time = get_non_idle_time();
    /* Subtract the accumulated time from the main time span. */
    int num_worker_threads = thread_instance::get_num_threads();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>
           (MYCLOCK::now() - main_timer->start);
    double total_main = time_span.count() *
                fmin(hardware_concurrency(), num_worker_threads);
    double elapsed = total_main - non_idle_time;
    double rate = elapsed > 0.0 ? ((elapsed/total_main)) : 0.0;
    profile * theprofile = new profile(rate, 0, NULL, false);
    {
        std::unique_lock<std::mutex> l(free_profile_set_mutex);
        free_profiles.insert(theprofile);
    }
    return theprofile;
  }

  /* Return the requested profile object to the user.
   * Return nullptr if doesn't exist. */
  profile * profiler_listener::get_profile(task_identifier &id) {
    /* Maybe we aren't processing profiler objects yet? Fire off a request. */
#ifdef APEX_HAVE_HPX
      // schedule an HPX action
    apex_schedule_process_profiles();
#else
    queue_signal.post();
#endif
    if (id.name == string(APEX_IDLE_RATE)) {
        return get_idle_rate();
    } else if (id.name == string(APEX_IDLE_TIME)) {
        return get_idle_time();
    } else if (id.name == string(APEX_NON_IDLE_TIME)) {
        profile * theprofile = new profile(get_non_idle_time(), 0, NULL, false);
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
#ifdef APEX_WITH_JUPYTER_SUPPORT
    // restart the main timer
    main_timer = std::make_shared<profiler>(task_wrapper::get_apex_main_wrapper());
#endif
  }

  /* After the consumer thread pulls a profiler off of the queue,
   * process it by updating its profile object in the map of profiles. */
  // TODO The name-based timer and address-based timer paths through
  // the code involve a lot of duplication -- this should be refactored
  // to remove the duplication so it's easier to maintain.
  unsigned int profiler_listener::process_profile(std::shared_ptr<profiler> &p, unsigned int tid)
  {
    if(p == nullptr) return 0;
    profile * theprofile;
    if(p->is_reset == reset_type::ALL) {
        reset_all();
        return 0;
    }
    double values[8] = {0};
    double tmp_num_counters = 0;
#if APEX_HAVE_PAPI
    tmp_num_counters = num_papi_counters;
    for (int i = 0 ; i < num_papi_counters ; i++) {
        if (p->papi_stop_values[i] > p->papi_start_values[i]) {
            values[i] = p->papi_stop_values[i] - p->papi_start_values[i];
        } else {
            values[i] = 0.0;
        }
    }
#endif
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    unordered_map<task_identifier, profile*>::const_iterator it = task_map.find(*(p->get_task_id()));
    if (it != task_map.end()) {
          // A profile for this ID already exists.
        theprofile = (*it).second;
        task_map_lock.unlock();
        if(p->is_reset == reset_type::CURRENT) {
            theprofile->reset();
        } else {
            theprofile->increment(p->elapsed(), tmp_num_counters, values, p->is_resume);
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
                it2 = throttled_tasks.find(*(p->get_task_id()));
              }
              if (it2 == throttled_tasks.end()) {
                  // lock the set for insert
                  {
                        write_lock_type l(throttled_event_set_mutex);
                      // was it inserted when we were waiting?
                      it2 = throttled_tasks.find(*(p->get_task_id()));
                      // no? OK - insert it.
                      if (it2 == throttled_tasks.end()) {
                          throttled_tasks.insert(*(p->get_task_id()));
                      }
                  }
                  if (apex_options::use_screen_output()) {
                      cout << "APEX: disabling lightweight timer "
                           << p->get_task_id()->get_name()
                            << endl;
                      fflush(stdout);
                  }
              }
          }
        }
#endif
      } else {
        // Create a new profile for this name.
        theprofile = new profile(p->is_reset == reset_type::CURRENT ? 0.0 : p->elapsed(), tmp_num_counters, values, p->is_resume, p->is_counter ? APEX_COUNTER : APEX_TIMER);
        task_map[*(p->get_task_id())] = theprofile;
        task_map_lock.unlock();
#ifdef APEX_HAVE_HPX
#ifdef APEX_REGISTER_HPX3_COUNTERS
        if(!_done) {
            if(get_hpx_runtime_ptr() != nullptr && p->get_task_id()->has_name()) {
                std::string timer_name(p->get_task_id()->get_name());
                //Don't register timers containing "/"
                if(timer_name.find("/") == std::string::npos) {
                    hpx::performance_counters::install_counter_type(
                    std::string("/apex/") + timer_name,
                    [p](bool r)->boost::int64_t{
                        boost::int64_t value(p->elapsed());
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
#if !defined(_MSC_VER)
      /* write the sample to the file */
      if (apex_options::task_scatterplot()) {
        if (!p->is_counter) {
            static int thresh = RAND_MAX/100;
            if (std::rand() < thresh) {
                /* before calling p->get_task_id()->get_name(), make sure we create
                 * a thread_instance object that is NOT a worker. */
                thread_instance::instance(false);
                std::unique_lock<std::mutex> task_map_lock(_mtx);
                task_scatterplot_samples << p->normalized_timestamp() << " "
                            << p->elapsed()*profiler::get_cpu_mhz()*1000000 << " "
                            << "'" << p->get_task_id()->get_name() << "'" << endl;
                int loc0 = task_scatterplot_samples.tellp();
                if (loc0 > 32768) {
                    // lock access to the file
                    // write using low-level file locking!
                    struct flock fl;
                    fl.l_type   = F_WRLCK;  /* F_RDLCK, F_WRLCK, F_UNLCK    */
                    fl.l_whence = SEEK_SET; /* SEEK_SET, SEEK_CUR, SEEK_END */
                    fl.l_start  = 0;        /* Offset from l_whence         */
                    fl.l_len    = 0;        /* length, 0 = to EOF           */
                    fl.l_pid    = getpid();      /* our PID                      */
                    fcntl(task_scatterplot_sample_file, F_SETLKW, &fl);  /* F_GETLK, F_SETLK, F_SETLKW */
                    // flush the string stream to the file
                    //lseek(task_scatterplot_sample_file, 0, SEEK_END);
                    ssize_t bytes_written = write(task_scatterplot_sample_file,
                          task_scatterplot_samples.str().c_str(), loc0);
                    if (bytes_written < 0) {
                        int errsv = errno;
                        perror("Error writing to scatterplot!");
                        fprintf(stderr, "Error writing scatterplot:\n%s\n",
                                strerror(errsv));
                    }
                    fl.l_type   = F_UNLCK;   /* tell it to unlock the region */
                    fcntl(task_scatterplot_sample_file, F_SETLK, &fl); /* set the region to unlocked */
                    // reset the stringstream
                    task_scatterplot_samples.str("");
                }
            }
        }
      }
#endif
    return 1;
  }

  inline unsigned int profiler_listener::process_dependency(task_dependency* td)
  {
      unordered_map<task_identifier, unordered_map<task_identifier, int>* >::const_iterator it = task_dependencies.find(td->parent);
      unordered_map<task_identifier, int> * depend;
      // if this is a new dependency for this parent?
      if (it == task_dependencies.end()) {
          depend = new unordered_map<task_identifier, int>();
          (*depend)[td->child] = 1;
          task_dependencies[td->parent] = depend;
      // otherwise, see if this parent has seen this child
      } else {
          depend = it->second;
          unordered_map<task_identifier, int>::const_iterator it2 = depend->find(td->child);
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
      size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
      unique_ptr<char[]> buf( new char[ size ] );
      snprintf( buf.get(), size, format.c_str(), args ... );
      return string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
  }

  void profiler_listener::write_one_timer(task_identifier &task_id,
          profile * p, stringstream &screen_output,
          stringstream &csv_output, double &total_accumulated,
          double &total_main, bool timer) {
      /* before calling task_id.get_name(), make sure we create
       * a thread_instance object that is NOT a worker. */
      thread_instance::instance(false);
      string action_name = task_id.get_name();
      if (action_name.compare(APEX_MAIN) == 0) {
          return; // don't write out apex main timer
      }
      string shorter(action_name);
      size_t maxlength = 30;
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
          screen_output << string_format("%30s", shorter.c_str()) << " : ";
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
            screen_output << "DISABLED (high frequency, short duration)" << endl;
            return;
        }
      }
#endif
      if(p->get_calls() < 1) {
        p->get_profile()->calls = 1;
      }
      if (p->get_calls() < 999999) {
          screen_output << string_format(PAD_WITH_SPACES, to_string((int)p->get_calls()).c_str()) << "   " ;
      } else {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_calls()) << "   " ;
      }
      if (p->get_type() == APEX_TIMER) {
        csv_output << "\"" << action_name << "\",";
        csv_output << llround(p->get_calls()) << ",";
        // convert MHz to Hz
        csv_output << std::llround(p->get_accumulated()) << ",";
        // convert MHz to microseconds
        csv_output << std::llround(p->get_accumulated()*profiler::get_cpu_mhz()*1000000);
        //screen_output << " --n/a--   " ;
        screen_output << string_format(FORMAT_SCIENTIFIC, (p->get_mean()*profiler::get_cpu_mhz())) << "   " ;
        //screen_output << " --n/a--   " ;
        screen_output << string_format(FORMAT_SCIENTIFIC, (p->get_accumulated()*profiler::get_cpu_mhz())) << "   " ;
        //screen_output << " --n/a--   " ;
        if (task_id.get_name().compare(APEX_MAIN) == 0) {
            screen_output << string_format(FORMAT_PERCENT, 100.0);
        } else {
            total_accumulated += p->get_accumulated();
            double tmp = ((p->get_accumulated()*profiler::get_cpu_mhz())/total_main)*100.0;
            if (tmp > 100.0) {
                screen_output << " --n/a--   " ;
            } else {
                screen_output << string_format(FORMAT_PERCENT, tmp);
            }
        }
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < num_papi_counters ; i++) {
            screen_output  << "   " << string_format(FORMAT_SCIENTIFIC, (p->get_papi_metrics()[i]));
            csv_output << "," << std::llround(p->get_papi_metrics()[i]);
        }
#endif
        screen_output << endl;
        csv_output << endl;
      } else {
        if (action_name.find('%') == string::npos) {
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_minimum()) << "   " ;
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_mean()) << "   " ;
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_maximum()) << "   " ;
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_accumulated()) << "   " ;
          screen_output << string_format(FORMAT_SCIENTIFIC, p->get_stddev()) << "   " ;
        } else {
          screen_output << string_format(FORMAT_PERCENT, p->get_minimum()) << "   " ;
          screen_output << string_format(FORMAT_PERCENT, p->get_mean()) << "   " ;
          screen_output << string_format(FORMAT_PERCENT, p->get_maximum()) << "   " ;
          screen_output << string_format(FORMAT_PERCENT, p->get_accumulated()) << "   " ;
          screen_output << string_format(FORMAT_PERCENT, p->get_stddev()) << "   " ;
        }
        //screen_output << " --n/a-- "  << endl;
        screen_output << endl;
      }
  }

  /* At program termination, write the measurements to the screen, or to CSV file, or both. */
  void profiler_listener::finalize_profiles(dump_event_data &data) {
    if (apex_options::use_tau()) {
      Tau_start("profiler_listener::finalize_profiles");
    }
    // our TOTAL available time is the elapsed * the number of threads, or cores
    int num_worker_threads = thread_instance::get_num_threads();
    task_identifier main_id(APEX_MAIN);
    profile * total_time = get_profile(main_id);
    double wall_clock_main = total_time->get_accumulated() * profiler::get_cpu_mhz();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_hpx_threads = 0;
    double total_main = wall_clock_main *
        fmin(hardware_concurrency(), num_worker_threads);
    // create a stringstream to hold all the screen output - we may not
    // want to write it out
    stringstream screen_output;
    // create a stringstream to hold all the CSV output - we may not
    // want to write it out
    stringstream csv_output;
    // iterate over the profiles in the address map
    screen_output << endl << "Elapsed time: " << wall_clock_main << " seconds" << endl;
    screen_output << "Cores detected: " << hardware_concurrency() << endl;
    screen_output << "Worker Threads observed: " << num_worker_threads << endl;
    screen_output << "Available CPU time: " << total_main << " seconds" << endl << endl;
    map<apex_function_address, profile*>::const_iterator it;
#if APEX_HAVE_PAPI
    for (int i = 0 ; i < num_papi_counters ; i++) {
       csv_output << ",\"" << metric_names[i] << "\"";
    }
#endif
    csv_output << endl;
    double total_accumulated = 0.0;
    unordered_map<task_identifier, profile*>::const_iterator it2;
    std::vector<task_identifier> id_vector;
    // iterate over the counters, and sort their names
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
        task_identifier task_id = it2->first;
        profile * p = it2->second;
        if (p->get_type() != APEX_TIMER) {
            id_vector.push_back(task_id);
        }
    }
    if (id_vector.size() > 0) {
        screen_output << "Counter                        : #samples |  minimum |    mean  |  maximum |   total  |  stddev " << endl;
        screen_output << "------------------------------------------------------------------------------------------------" << endl;
        std::sort(id_vector.begin(), id_vector.end());
        // iterate over the counters
        for(task_identifier task_id : id_vector) {
            profile * p = task_map[task_id];
            if (p) {
                write_one_timer(task_id, p, screen_output, csv_output, total_accumulated, total_main, false);
            }
        }
        screen_output << "------------------------------------------------------------------------------------------------" << endl << endl;;
    }
    csv_output << "\"task\",\"num calls\",\"total cycles\",\"total microseconds\"";
    screen_output << "Timer                                                :  #calls  |    mean  |   total  |  % total  " << std::regex_replace(apex_options::papi_metrics(), std::regex("PAPI_"), "| ") << endl;
    screen_output << "------------------------------------------------------------------------------------------------" << endl;
     id_vector.clear();
    // iterate over the timers
    for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
        profile * p = it2->second;
        task_identifier task_id = it2->first;
        if (p->get_type() == APEX_TIMER) {
            id_vector.push_back(task_id);
        }
    }
    // sort by name
    std::sort(id_vector.begin(), id_vector.end());
    // iterate over the timers
    for(task_identifier task_id : id_vector) {
        profile * p = task_map[task_id];
        if (p) {
            write_one_timer(task_id, p, screen_output, csv_output, total_accumulated, total_main, true);
            if (task_id.get_name().compare(APEX_MAIN) != 0) {
                total_hpx_threads = total_hpx_threads + p->get_calls();
            }
        }
    }
    double idle_rate = total_main - (total_accumulated*profiler::get_cpu_mhz());
    if (idle_rate >= 0.0) {
      screen_output << string_format("%52s", APEX_IDLE_TIME) << " : ";
      screen_output << "                      "; // pad with spaces for #calls, mean
      screen_output << string_format(FORMAT_SCIENTIFIC, idle_rate) << "   " ;
      screen_output << string_format(FORMAT_PERCENT, ((idle_rate/total_main)*100)) << endl;
    }
    screen_output << "------------------------------------------------------------------------------------------------" << endl;
    screen_output << string_format("%52s", "Total timers") << " : ";
    //if (total_hpx_threads < 999999) {
        //screen_output << string_format(PAD_WITH_SPACES, to_string((int)(total_hpx_threads))) << std::endl;
    //} else {
    std::stringstream total_ss;
    total_ss.imbue(std::locale(""));
    total_ss << std::fixed << ((uint64_t)total_hpx_threads);
        screen_output << total_ss.str() << std::endl;
    //}
    if (apex_options::use_screen_output()) {
        cout << screen_output.str();
        data.output = screen_output.str();
    }
    if (apex_options::use_csv_output()) {
        ofstream csvfile;
        stringstream csvname;
        csvname << apex_options::output_file_path();
        csvname << filesystem_separator() << "apex." << node_id << ".csv";
        csvfile.open(csvname.str(), ios::out);
        csvfile << csv_output.str();
        csvfile.close();
    }
    if (apex_options::use_tau()) {
      Tau_stop("profiler_listener::finalize_profiles");
    }
  }

/* The following code is from:
   http://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale */
class node_color {
public:
    double red;
    double green;
    double blue;
    node_color() : red(1.0), green(1.0), blue(1.0) {}
    int convert(double in) { return (int)(in * 255.0); }
} ;

node_color * get_node_color_visible(double v, double vmin, double vmax) {
   node_color * c = new node_color();
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;
   // Green should be full on.
   c->blue = 1.0;
   // red and green should increase as the fraction decreases.
   double fraction = 1.0 - ( (v - vmin) / dv );
   c->red = 0.10 + (0.90 * fraction);
   c->green = 0.75 + (0.25 * fraction);
   return c;
}

node_color * get_node_color(double v,double vmin,double vmax)
{
   node_color * c = new node_color();
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c->red = 0;
      c->green = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c->red = 0;
      c->blue = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c->red = 4 * (v - vmin - 0.5 * dv) / dv;
      c->blue = 0;
   } else {
      c->green = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c->blue = 0;
   }

   return(c);
}

  void profiler_listener::write_taskgraph(void) {
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

    myfile << "digraph prof {\n rankdir=\"LR\";\n node [shape=box];\n";
    for(auto dep = task_dependencies.begin(); dep != task_dependencies.end(); dep++) {
        task_identifier parent = dep->first;
        auto children = dep->second;
        string parent_name = parent.get_name();
        for(auto offspring = children->begin(); offspring != children->end(); offspring++) {
            task_identifier child = offspring->first;
            int count = offspring->second;
            string child_name = child.get_name();
            myfile << "  \"" << parent_name << "\" -> \"" << child_name << "\"";
            myfile << " [ label=\"  count: " << count << "\" ]; " << std::endl;
        }
    }
    // delete the dependency graph
    for(auto dep = task_dependencies.begin(); dep != task_dependencies.end(); dep++) {
        auto children = dep->second;
        children->clear();
        delete(children);
    }
    task_dependencies.clear();

    // our TOTAL available time is the elapsed * the number of threads, or cores
    /*
    int num_worker_threads = thread_instance::get_num_threads();
#ifdef APEX_HAVE_HPX
    num_worker_threads = num_worker_threads - num_non_worker_threads_registered;
#endif
    double total_main = main_timer->elapsed() * fmin(hardware_concurrency(), num_worker_threads);
    */

    // output nodes with  "main" [shape=box; style=filled; fillcolor="#ff0000" ];
    unordered_map<task_identifier, profile*>::const_iterator it;
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it = task_map.begin(); it != task_map.end(); it++) {
      profile * p = it->second;
      // shouldn't happen, but?
      if (p == nullptr) continue;
      if (p->get_type() == APEX_TIMER) {
        node_color * c = get_node_color_visible(p->get_mean(), 0.0, main_timer->elapsed());
        task_identifier task_id = it->first;
        myfile << "  \"" << task_id.get_name() << "\" [shape=box; style=filled; fillcolor=\"#" <<
            setfill('0') << setw(2) << hex << c->convert(c->red) <<
            setfill('0') << setw(2) << hex << c->convert(c->green) <<
            setfill('0') << setw(2) << hex << c->convert(c->blue) << "\"" <<
            "; label=\"" << task_id.get_name() << ":\\n(" << 
            (p->get_calls()) << ") " <<
            (p->get_mean()*profiler::get_cpu_mhz()) << "s\" ];" << std::endl;
        delete(c);
      }
    }
    myfile << "}\n";
    myfile.close();
  }

  /* When writing a TAU profile, write out a timer line */
  void format_line(ofstream &myfile, profile * p) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << ((p->get_accumulated()*profiler::get_cpu_mhz())) << " ";
    myfile << ((p->get_accumulated()*profiler::get_cpu_mhz())) << " ";
    myfile << 0 << " ";
    myfile << "GROUP=\"TAU_USER\" ";
    myfile << endl;
  }

  /* When writing a TAU profile, write out the main timer line */
  void format_line(ofstream &myfile, profile * p, double not_main) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << (max(((p->get_accumulated()*profiler::get_cpu_mhz()) - not_main),0.0)) << " ";
    myfile << ((p->get_accumulated()*profiler::get_cpu_mhz())) << " ";
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
    std::unique_lock<std::mutex> task_map_lock(_task_map_mutex);
    for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == APEX_COUNTER) {
        counter_events++;
      }
    }
    int function_count = task_map.size() - counter_events;

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
    for(it2 = task_map.begin(); it2 != task_map.end(); it2++) {
      profile * p = it2->second;
      task_identifier task_id = it2->first;
      if(p->get_type() == APEX_TIMER) {
        string action_name = task_id.get_name();
        if(strcmp(action_name.c_str(), APEX_MAIN) == 0) {
          mainp = p;
        } else {
          myfile << "\"" << action_name << "\" ";
          format_line (myfile, p);
          not_main += (p->get_accumulated()*profiler::get_cpu_mhz()*1000000);
        }
      }
    }
    if (mainp != nullptr) {
      myfile << "\"" << APEX_MAIN << "\" ";
      format_line (myfile, mainp, not_main);
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
      thread_instance::delete_instance();
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
#else
              pl->process_profiles();
#endif
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
      Tau_start("profiler_listener::process_profiles");
    }

    std::shared_ptr<profiler> p;
    task_dependency* td;
#ifdef APEX_HAVE_HPX
    bool schedule_another_task = false;
    {
        std::unique_lock<std::mutex> queue_lock(queue_mtx);
        for (auto a_queue : allqueues) {
            int i = 0;
            while(!_done && a_queue->try_dequeue(p)) {
                process_profile(p, 0);
                if (++i > 1000) {
                    schedule_another_task = true;
                    break;
                }
            }
        }
    }
    if (apex_options::use_taskgraph_output()) {
        std::unique_lock<std::mutex> queue_lock(queue_mtx);
        for (auto a_queue : dependency_queues) {
            int i = 0;
            while(!_done && a_queue->try_dequeue(td)) {
                process_dependency(td);
                if (++i > 1000) {
                    schedule_another_task = true;
                    break;
                }
            }
        }
    }
#else
    // Main loop. Stay in this loop unless "done".
    while (!_done) {
        queue_signal.wait();
        if (apex_options::use_tau()) {
            Tau_start("profiler_listener::process_profiles: main loop");
        }
        { 
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            for (auto a_queue : allqueues) {
                while(!_done && a_queue->try_dequeue(p)) {
                    process_profile(p, 0);
                }
            }
        }
        if (apex_options::use_taskgraph_output()) {
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            for (auto a_queue : dependency_queues) {
                while(!_done && a_queue->try_dequeue(td)) {
                    process_dependency(td);
                }
            }
        }
        if (apex_options::use_tau()) {
            Tau_stop("profiler_listener::process_profiles: main loop");
        }
        // release the flag, and wait for the queue_signal
        consumer_task_running.clear(memory_order_release);
    }
#endif

#ifdef APEX_HAVE_HPX // don't hang out in this task too long.
    if (schedule_another_task) {
        apex_schedule_process_profiles();
    }
#endif

    if (apex_options::use_tau()) {
      Tau_stop("profiler_listener::process_profiles");
    }
  }

#if APEX_HAVE_PAPI
APEX_NATIVE_TLS int EventSet = PAPI_NULL;
enum papi_state { papi_running, papi_suspended };
APEX_NATIVE_TLS papi_state thread_papi_state = papi_suspended;
#define PAPI_ERROR_CHECK(name) \
if (rc != 0) cout << "PAPI error! " << name << ": " << PAPI_strerror(rc) << endl;

  void profiler_listener::initialize_PAPI(bool first_time) {
      int rc = 0;
      if (first_time) {
        PAPI_library_init( PAPI_VER_CURRENT );
        //rc = PAPI_multiplex_init(); // use more counters than allowed
        //PAPI_ERROR_CHECK("PAPI_multiplex_init");
        PAPI_thread_init( &thread_instance::get_id );
        // default
        //rc = PAPI_set_domain(PAPI_DOM_ALL);
        //PAPI_ERROR_CHECK("PAPI_set_domain");
      } else {
        PAPI_register_thread();
      }
      rc = PAPI_create_eventset(&EventSet);
      PAPI_ERROR_CHECK("PAPI_create_eventset");
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
      if (strlen(apex_options::papi_metrics()) > 0) {
        std::stringstream tmpstr(apex_options::papi_metrics());
        // use stream iterators to copy the stream to the vector as whitespace separated strings
        std::istream_iterator<std::string> tmpstr_it(tmpstr);
        std::istream_iterator<std::string> tmpstr_end;
        std::vector<std::string> tmpstr_results(tmpstr_it, tmpstr_end);
        int code;
        // iterate over the counter names in the vector
        for (auto p : tmpstr_results) {
          int rc = PAPI_event_name_to_code(const_cast<char*>(p.c_str()), &code);
          if (PAPI_query_event (code) == PAPI_OK) {
            rc = PAPI_add_event(EventSet, code);
            PAPI_ERROR_CHECK("PAPI_add_event");
            if (rc != 0) { printf ("Event that failed: %s\n", p.c_str()); }
            if (first_time) {
              metric_names.push_back(string(p.c_str()));
              num_papi_counters++;
            }
          }
        }
        if (!apex_options::papi_suspend()) {
            rc = PAPI_start( EventSet );
            PAPI_ERROR_CHECK("PAPI_start");
            thread_papi_state = papi_running;
        }
      }
  }

#endif

  /* When APEX gets a STARTUP event, do some initialization. */
  void profiler_listener::on_startup(startup_event_data &data) {
    if (!_done) {
      my_tid = (unsigned int)thread_instance::get_id();
      async_thread_setup();
#ifndef APEX_HAVE_HPX
      // Start the consumer thread, to process profiler objects.
      consumer_thread = new std::thread(consumer_process_profiles_wrapper);
#endif

#if APEX_HAVE_PAPI
      initialize_PAPI(true);
      event_sets[0] = EventSet;
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
        << ", priority=" << param.sched_priority << " of " << sched_get_priority_min(policy) << "," << sched_get_priority_max(policy)  << std::endl;
      //param.sched_priority = 10;
      if ((retcode = pthread_setschedparam(threadID, policy, &param)) != 0)
      {
        errno = retcode;
        perror("pthread_setschedparam");
        exit(EXIT_FAILURE);
      }
#endif

      // time the whole application.
      main_timer = std::make_shared<profiler>(task_wrapper::get_apex_main_wrapper());
#if APEX_HAVE_PAPI
      if (num_papi_counters > 0 && !apex_options::papi_suspend() && thread_papi_state == papi_running) {
        int rc = PAPI_read( EventSet, main_timer->papi_start_values );
        PAPI_ERROR_CHECK("PAPI_read");
      }
#endif
    }
    node_id = data.comm_rank;
  }

  /* On the dump event, output all the profiles regardless of whether
   * the screen dump flag is set. */
  void profiler_listener::on_dump(dump_event_data &data) {
    if (_done) { return; }

    // wait until any other threads are done processing dependencies
    while(consumer_task_running.test_and_set(memory_order_acq_rel)) { }

      // stop the main timer, and process that profile?
      main_timer->stop();
      push_profiler((unsigned int)thread_instance::get_id(), main_timer);

      // output to screen?
      if ((apex_options::use_screen_output() ||
           apex_options::use_taskgraph_output() ||
           apex_options::use_csv_output()) && node_id == 0)
      {
        size_t ignored = 0;
        { // we need to lock in case another thread appears
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            for (auto a_queue : allqueues) {
                ignored += a_queue->size_approx();
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
            std::unique_lock<std::mutex> queue_lock(queue_mtx);
            for (unsigned int i=0; i<allqueues.size(); ++i) {
                if (apex_options::use_tau()) {
                    Tau_start("profiler_listener::concurrent_cleanup");
                }
                concurrent_cleanup(i);
                if (apex_options::use_tau()) {
                    Tau_stop("profiler_listener::concurrent_cleanup");
                }
            }
        }
        if (ignored > 100000) {
          std::cerr << "done." << std::endl;
        }
        finalize_profiles(data);
      }
      if (apex_options::use_taskgraph_output() && node_id == 0)
      {
        write_taskgraph();
      }

      // output to 1 TAU profile per process?
      if (apex_options::use_profile_output() && !apex_options::use_tau()) {
        write_profile();
      }
#if !defined(_MSC_VER)
      if (apex_options::task_scatterplot()) {
          // get the length of the stream
          int loc0 = task_scatterplot_samples.tellp();
          // lock access to the file
          // write using low-level file locking!
          struct flock fl;
          fl.l_type   = F_WRLCK;  /* F_RDLCK, F_WRLCK, F_UNLCK    */
          fl.l_whence = SEEK_SET; /* SEEK_SET, SEEK_CUR, SEEK_END */
          fl.l_start  = 0;        /* Offset from l_whence         */
          fl.l_len    = 0;        /* length, 0 = to EOF           */
          fl.l_pid    = getpid();      /* our PID                      */
          fcntl(task_scatterplot_sample_file, F_SETLKW, &fl);  /* F_GETLK, F_SETLK, F_SETLKW */
          // flush the string stream to the file
          //lseek(task_scatterplot_sample_file, 0, SEEK_END);
          ssize_t bytes_written = write(task_scatterplot_sample_file,
                task_scatterplot_samples.str().c_str(), loc0);
          if (bytes_written < 0) {
              int errsv = errno;
              perror("Error writing to scatterplot!");
              fprintf(stderr, "Error writing scatterplot:\n%s\n",
                      strerror(errsv));
          }
          fl.l_type   = F_UNLCK;   /* tell it to unlock the region */
          fcntl(task_scatterplot_sample_file, F_SETLK, &fl); /* set the region to unlocked */
          close(task_scatterplot_sample_file);
      }
#endif
      // restart the main timer
      main_timer = std::make_shared<profiler>(task_wrapper::get_apex_main_wrapper());
      if (data.reset) {
          reset_all();
      }
      // on_dump() releasing the "task_running" flag
      consumer_task_running.clear(memory_order_release);
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
    if (_done) { return; }
    if (!_done) {
      _done = true;
      //node_id = data.node_id;
      //sleep(1);
#ifndef APEX_HAVE_HPX
      queue_signal.post();
      queue_signal.dump_stats();
      if (consumer_thread != nullptr) {
          queue_signal.post(); // one more time, just to be sure
          consumer_thread->join();
      }
#endif

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
      my_tid = (unsigned int)thread_instance::get_id();
      async_thread_setup();
#if APEX_HAVE_PAPI
      initialize_PAPI(false);
      event_set_mutex.lock();
      if (my_tid >= event_sets.size()) {
        if (my_tid >= event_sets.size()) {
          event_sets.resize(my_tid + 1);
        }
      }
      event_sets[my_tid] = EventSet;
      event_set_mutex.unlock();
#endif
    }
    APEX_UNUSED(data);
  }

  extern "C" int main (int, char**);

  /* When a start event happens, create a profiler object. Unless this
   * named event is throttled, in which case do nothing, as quickly as possible */
  inline bool profiler_listener::_common_start(std::shared_ptr<task_wrapper> &tt_ptr, bool is_resume) {
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
            * The throw is removed, because it is a performance penalty on some systems
            * on_start now returns a boolean
            */
            //throw disabled_profiler_exception(); // to be caught by apex::start/resume
            return false;
        }
      }
#endif
      // start the profiler object, which starts our timers
      //std::shared_ptr<profiler> p = std::make_shared<profiler>(tt_ptr, is_resume);
      // get the right task identifier, based on whether there are aliases
      profiler * p = new profiler(tt_ptr, is_resume);
      p->guid = thread_instance::get_guid();
      thread_instance::instance().set_current_profiler(p);
#if APEX_HAVE_PAPI
      if (num_papi_counters > 0 && !apex_options::papi_suspend()) {
          // if papi was previously suspended, we need to start the counters
          if (thread_papi_state == papi_suspended) {
            int rc = PAPI_start( EventSet );
            PAPI_ERROR_CHECK("PAPI_start");
            thread_papi_state = papi_running;
          }
          int rc = PAPI_read( EventSet, p->papi_start_values );
          PAPI_ERROR_CHECK("PAPI_read");
      } else {
          // if papi is still running, stop the counters
          if (thread_papi_state == papi_running) {
            long long dummy[8];
            int rc = PAPI_stop( EventSet, dummy );
            PAPI_ERROR_CHECK("PAPI_stop");
            thread_papi_state = papi_suspended;
          }
      }
#endif
    } else {
        return false;
    }
    return true;
  }

  inline void profiler_listener::push_profiler(int my_tid, std::shared_ptr<profiler> &p) {
        // if we aren't processing profiler objects, just return.
        if (!apex_options::process_async_state()) { return; }
#ifdef APEX_TRACE_APEX
        if (p->get_task_id()->name == "apex::process_profiles") { return; }
#endif
      // we have to make a local copy, because lockfree queues DO NOT SUPPORT shared_ptrs!
      thequeue()->enqueue(p);

#ifndef APEX_HAVE_HPX
      // Check to see if the consumer is already running, to avoid calling "post"
      // too frequently - it is rather costly.
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
  inline void profiler_listener::_common_stop(std::shared_ptr<profiler> &p, bool is_yield) {
    if (!_done) {
      if (p) {
        p->stop(is_yield);
#if APEX_HAVE_PAPI
        if (num_papi_counters > 0 && !apex_options::papi_suspend() && thread_papi_state == papi_running) {
            int rc = PAPI_read( EventSet, p->papi_stop_values );
            PAPI_ERROR_CHECK("PAPI_read");
        }
#endif
        push_profiler(my_tid, p);
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
      std::shared_ptr<profiler> p = std::make_shared<profiler>(task_identifier::get_task_id(*data.counter_name), data.counter_value);
      p->is_counter = data.is_counter;
      push_profiler(my_tid, p);
    }
  }

  void profiler_listener::on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
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
    task_identifier * parent = task_wrapper::get_apex_main_wrapper()->task_id;
    dependency_queue()->enqueue(new task_dependency(parent, id));
  }

  /* Communication send event. Save the number of bytes. */
  void profiler_listener::on_send(message_event_data &data) {
    if (!_done) {
      std::shared_ptr<profiler> p = std::make_shared<profiler>(task_identifier::get_task_id("Bytes Sent"), (double)data.size);
      push_profiler(0, p);
    }
  }

  /* Communication recv event. Save the number of bytes. */
  void profiler_listener::on_recv(message_event_data &data) {
    if (!_done) {
      std::shared_ptr<profiler> p = std::make_shared<profiler>(task_identifier::get_task_id("Bytes Received"), (double)data.size);
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
    std::shared_ptr<profiler> p;
    p = std::make_shared<profiler>(id, false, reset_type::CURRENT);
    push_profiler(my_tid, p);
  }

  profiler_listener::~profiler_listener (void) {
      _done = true; // yikes!
      finalize();
      delete_profiles();
#ifndef APEX_HAVE_HPX
#ifndef APEX_STATIC // unbelievable.  Deleting this object can crash in a static link.
      delete consumer_thread;
#endif
#endif
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

  };

}

#ifdef APEX_HAVE_HPX
HPX_DECLARE_ACTION(APEX_TOP_LEVEL_PACKAGE::profiler_listener::process_profiles_wrapper, apex_internal_process_profiles_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(apex_internal_process_profiles_action);
HPX_PLAIN_ACTION(APEX_TOP_LEVEL_PACKAGE::profiler_listener::process_profiles_wrapper, apex_internal_process_profiles_action);

void apex_schedule_process_profiles() {
    if(get_hpx_runtime_ptr() == nullptr) return;
    if(!thread_instance::is_worker()) return;
    if(hpx_shutdown) {
        APEX_TOP_LEVEL_PACKAGE::profiler_listener::process_profiles_wrapper();
    } else {
        if(!consumer_task_running.test_and_set(memory_order_acq_rel)) {
            apex_internal_process_profiles_action act;
            try {
                hpx::apply(act, hpx::find_here());
            } catch(...) {
                // During shutdown, we can't schedule a new task,
                // so we process profiles ourselves.
                profiler_listener::process_profiles_wrapper();
            }
        }
    }
}

#endif


