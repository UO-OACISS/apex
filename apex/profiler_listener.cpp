//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "profiler_listener.hpp"
#include "profiler.hpp"
#include "thread_instance.hpp"
#include "semaphore.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include "apex_options.hpp"

#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>
#include <unistd.h>
#include <sched.h>
#include <cstdio>

#if defined(APEX_THROTTLE)
#define APEX_THROTTLE_CALLS 1000  
#define APEX_THROTTLE_PERCALL 0.00001 // 10 microseconds. 
#include <unordered_set>
#endif

#if APEX_HAVE_BFD
#include "apex_bfd.h"
#endif

#if APEX_HAVE_PAPI
#include "papi.h"
#endif

//#define MAX_QUEUE_SIZE 1024*1024
#define MAX_QUEUE_SIZE 128
#define INITIAL_NUM_THREADS 2

#define APEX_MAIN "APEX MAIN"

using namespace std;

__thread unsigned int my_tid = 0; // the current thread's TID in APEX

namespace apex {

  /* This is the array of profiler queues, one for each worker thread. It
   * is initialized to a length of 8, there is code in on_new_thread() to
   * increment it if necessary.  */
  std::vector<boost::lockfree::spsc_queue<profiler*>* > profiler_queues(8);

#if APEX_HAVE_PAPI
  std::vector<int> event_sets(8);
#endif

  /* This is the thread that will read from the queues of profiler
   * objects, and update the profiles. */
  boost::thread * consumer_thread;

  /* Flag indicating that we are done inserting into the queues, so the
   * consumer knows when to exit */
  boost::atomic<bool> done (false);

  /* how the workers signal the consumer that there are new profiler objects
   * on the queue to consume */
  semaphore queue_signal;

  /* The profiles, mapped by name. Profiles will be in this map or the other
   * one, not both. It depends whether the timers are identified by name
   * or address. */
  static map<string, profile*> name_map;
  static vector<map<string, profile*>* > thread_name_maps(INITIAL_NUM_THREADS);

  /* The profiles, mapped by address */
  static map<void*, profile*> address_map;
  vector<map<void*, profile*>* > thread_address_maps(INITIAL_NUM_THREADS);

  /* If we are throttling, keep a set of addresses that should not be timed.
   * We also have a set of names. */
#if defined(APEX_THROTTLE)
  static unordered_set<void*> throttled_addresses;
  static unordered_set<string> throttled_names;
#endif

  /* measurement of entire application */
  profiler * profiler_listener::main_timer(NULL);

  /* the node id is needed for profile output. */
  int profiler_listener::node_id(0);

  /* A lock necessary for registering new threads */
  boost::mutex profiler_listener::_mtx ;

  /* Return the requested profile object to the user.
   * Return NULL if doesn't exist. */
  profile * profiler_listener::get_profile(apex_function_address address) {
    map<void*, profile*>::iterator it = address_map.find((void*)address);
    if (it != address_map.end()) {
      return (*it).second;
    }
    return NULL;
  }

  /* Return the requested profile object to the user.
   * Return NULL if doesn't exist. */
  profile * profiler_listener::get_profile(string &timer_name) {
    map<string, profile*>::iterator it = name_map.find(timer_name);
    if (it != name_map.end()) {
      return (*it).second;
    }
    return NULL;
  }

  /* forward declaration, defined below */
#if APEX_HAVE_BFD
  extern string * lookup_address(uintptr_t ip);
#endif

  /* After the consumer thread pulls a profiler off of the queue,
   * process it by updating its profile object in the map of profiles. */
  inline void profiler_listener::process_profile(profiler * p, unsigned int tid)
  {
    profile * theprofile;
    // Look for the profile object by name, if applicable
    if (p->have_name) {
      map<string, profile*>::iterator it = name_map.find(*(p->timer_name));
      if (it != name_map.end()) {
        // A profile for this name already exists.
        theprofile = (*it).second;
        theprofile->increment(p->elapsed());
#if defined(APEX_THROTTLE)
        // Is this a lightweight task? If so, we shouldn't measure it any more,
        // in order to reduce overhead.
        if (theprofile->get_calls() > APEX_THROTTLE_CALLS && 
            theprofile->get_mean() < APEX_THROTTLE_PERCALL) { 
          unordered_set<string>::iterator it = throttled_names.find(p->timer_name);
          if (it == throttled_names.end()) { 
            throttled_names.insert(p->timer_name);
            cout << "APEX Throttled " << p->timer_name << endl;
          }
        }
#endif
      } else {
        // Create a new profile for this name.
        theprofile = new profile(p->elapsed(), p->is_counter ? COUNTER : TIMER);
        name_map[*(p->timer_name)] = theprofile;
      }
      // now do thread-specific measurement.
      map<string, profile*>* the_map = thread_name_maps[tid];
      it = the_map->find(*(p->timer_name));
      if (it != the_map->end()) {
        // A profile for this name already exists.
        theprofile = (*it).second;
        theprofile->increment(p->elapsed());
      } else {
        // Create a new profile for this name.
        theprofile = new profile(p->elapsed(), p->is_counter ? COUNTER : TIMER);
        (*the_map)[*(p->timer_name)] = theprofile;
      }
    } else {
      map<void*, profile*>::const_iterator it2 = address_map.find(p->action_address);
      if (it2 != address_map.end()) {
        // A profile for this name already exists.
        theprofile = (*it2).second;
        theprofile->increment(p->elapsed());
#if defined(APEX_THROTTLE)
        // Is this a lightweight task? If so, we shouldn't measure it any more,
        // in order to reduce overhead.
        if (theprofile->get_calls() > APEX_THROTTLE_CALLS && 
            theprofile->get_mean() < APEX_THROTTLE_PERCALL) { 
          unordered_set<void*>::iterator it = throttled_addresses.find(p->action_address);
          if (it == throttled_addresses.end()) { 
            throttled_addresses.insert(p->action_address);
#if defined(HAVE_BFD)
            cout << "APEX Throttled " << *(lookup_address((uintptr_t)p->action_address)) << endl;
#else
            cout << "APEX Throttled " << p->action_address << endl;
#endif
          }
        }
#endif
      } else {
        // Create a new profile for this address.
        theprofile = new profile(p->elapsed());
        address_map[p->action_address] = theprofile;
      }
      // now do thread-specific measurement
      map<void*, profile*>* the_map = thread_address_maps[tid];
      it2 = the_map->find(p->action_address);
      if (it2 != the_map->end()) {
        // A profile for this name already exists.
        theprofile = (*it2).second;
        theprofile->increment(p->elapsed());
      } else {
        // Create a new profile for this address.
        theprofile = new profile(p->elapsed());
        (*the_map)[p->action_address] = theprofile;
      }
    }
    // done with the profiler object
    delete(p);
  }

  /* Cleaning up memory. Not really necessary, because it only gets
   * called at shutdown. But a good idea to do regardless. */
  void profiler_listener::delete_profiles(void) {
    // iterate over the map and free the objects in the map
    map<void*, profile*>::const_iterator it;
    for(it = address_map.begin(); it != address_map.end(); it++) {
      delete it->second;
    }
    // iterate over the map and free the objects in the map
    map<string, profile*>::const_iterator it2;
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      delete it2->second;
    }
    // clear the maps.
    address_map.clear();
    name_map.clear();
    // iterate over the queues, and delete them
    unsigned int i = 0;
    for (i = 0 ; i < profiler_queues.size(); i++) {
      if (profiler_queues[i]) {
        delete (profiler_queues[i]);
      }
    }
    // clear the vector of queues.
    profiler_queues.clear();

    // iterate over the vector of profile objects for per-thread measurement
    for (i = 0 ; i < thread_address_maps.size(); i++) {
      if (thread_address_maps[i]) {
        for(it = thread_address_maps[i]->begin(); it != thread_address_maps[i]->end(); it++) {
          delete it->second;
        }
        delete (thread_address_maps[i]);
      }
    }
    // clear the vector of maps.
    thread_address_maps.clear();

    // iterate over the vector of profile objects for per-thread measurement
    for (i = 0 ; i < thread_name_maps.size(); i++) {
      if (thread_name_maps[i]) {
        for(it2 = thread_name_maps[i]->begin(); it2 != thread_name_maps[i]->end(); it2++) {
          delete it2->second;
        }
        delete (thread_name_maps[i]);
      }
    }
    // clear the vector of maps.
    thread_name_maps.clear();
  }

  /* At program termination, write the measurements to the screen. */
  void profiler_listener::finalize_profiles(void) {
    // iterate over the profiles in the address map 
    map<void*, profile*>::const_iterator it;
    cout << "Action, #calls, min, mean, max, total, stddev" << endl;
    for(it = address_map.begin(); it != address_map.end(); it++) {
      profile * p = it->second;
      void * function_address = it->first;
#if defined(APEX_THROTTLE)
      // if this profile was throttled, don't output the measurements.
      // they are limited and bogus, anyway.
      unordered_set<void*>::const_iterator it = throttled_addresses.find(function_address);
      if (it != throttled_addresses.end()) { continue; }
#endif
#if APEX_HAVE_BFD
      // translate the address to a name
      string * tmp = lookup_address((uintptr_t)function_address);
      cout << "\"" << *tmp << "\", " ;
#else
      cout << "\"" << function_address << "\", " ;
#endif
      cout << p->get_calls() << ", " ;
      cout << p->get_minimum() << ", " ;
      cout << p->get_mean() << ", " ;
      cout << p->get_maximum() << ", " ;
      cout << p->get_accumulated() << ", " ;
      cout << p->get_stddev() << endl;
    }
    map<string, profile*>::const_iterator it2;
    // iterate over the profiles in the address map 
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      string action_name = it2->first;
#if defined(APEX_THROTTLE)
      // if this profile was throttled, don't output the measurements.
      // they are limited and bogus, anyway.
      unordered_set<void*>::const_iterator it = throttled_names.find(action_name);
      if (it != throttled_names.end()) { continue; }
#endif
      cout << "\"" << action_name << "\", " ;
      cout << p->get_calls() << ", " ;
      cout << p->get_minimum() << ", " ;
      cout << p->get_mean() << ", " ;
      cout << p->get_maximum() << ", " ;
      cout << p->get_accumulated() << ", " ;
      cout << p->get_stddev() << endl;
    }
  }

  /* When writing a TAU profile, write out a timer line */
  void format_line(ofstream &myfile, profile * p) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << (p->get_accumulated() * 1000000.0) << " ";
    myfile << (p->get_accumulated() * 1000000.0) << " ";
    myfile << 0 << " ";
    myfile << "GROUP=\"TAU_USER\" ";
    myfile << endl;
  }

  /* When writing a TAU profile, write out the main timer line */
  void format_line(ofstream &myfile, profile * p, double not_main) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << (max((p->get_accumulated() - not_main),0.0) * 1000000.0) << " ";
    myfile << (p->get_accumulated() * 1000000.0) << " ";
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
  void profiler_listener::write_profile(unsigned int tid) {
    ofstream myfile;
    stringstream datname;

    if (thread_name_maps[tid] == NULL || thread_address_maps[tid] == NULL) return;
    map<string, profile*>* the_name_map = thread_name_maps[tid];
    map<void*, profile*>* the_address_map = thread_address_maps[tid];

    // name format: profile.nodeid.contextid.threadid
    // We only write one profile per process (for now).
    datname << "profile." << node_id << ".0." << tid;
    myfile.open(datname.str().c_str());
    int counter_events = 0;

    // Determine number of counter events, as these need to be
    // excluded from the number of normal timers
    map<string, profile*>::const_iterator it2;
    for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == COUNTER) {
        counter_events++;
      }
    }
    int function_count = the_address_map->size() + (the_name_map->size() - counter_events);

    // Print the normal timers to the profile file
    // 1504 templated_functions_MULTI_TIME
    myfile << function_count << " templated_functions_MULTI_TIME" << endl;
    // # Name Calls Subrs Excl Incl ProfileCalls #
    myfile << "# Name Calls Subrs Excl Incl ProfileCalls #" << endl;
    thread_instance ti = thread_instance::instance();

    // Iterate over the profiles which are associated to a function
    // by address. All of these are regular timers.
    map<void*, profile*>::const_iterator it;
    for(it = the_address_map->begin(); it != the_address_map->end(); it++) {
      profile * p = it->second;
      // ".TAU application" 1 8 8658984 8660739 0 GROUP="TAU_USER"
      void * function_address = it->first;
#if APEX_HAVE_BFD
      string * tmp = lookup_address((uintptr_t)function_address);
      myfile << "\"" << *tmp << "\" ";
#else
      myfile << "\"" << ti.map_addr_to_name(function_address) << "\" ";
#endif
      format_line (myfile, p);
    }                                              

    // Iterate over the profiles which are associated to a function
    // by name. Only output the regular timers now. Counters are
    // in a separate section, below.
	profile * mainp = NULL;
	double not_main = 0.0;
    for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == TIMER) {
        string action_name = it2->first;
	    if(strcmp(action_name.c_str(), APEX_MAIN) == 0) {
		  mainp = p;
		} else {
          myfile << "\"" << action_name << "\" ";
          format_line (myfile, p);
		  not_main += p->get_accumulated();
        }
      }
    }
	if (mainp != NULL) {
      myfile << "\"" << APEX_MAIN << "\" ";
      format_line (myfile, mainp, not_main);
	}

    // 0 aggregates
    myfile << "0 aggregates" << endl;

    // Now process the counters, if there are any.
    if(counter_events > 0) {
      myfile << counter_events << " userevents" << endl;
      myfile << "# eventname numevents max min mean sumsqr" << endl;
      for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
        profile * p = it2->second;
        if(p->get_type() == COUNTER) {
          string action_name = it2->first;
          myfile << "\"" << action_name << "\" ";
          format_counter_line (myfile, p);
        }
      }
    }
    myfile.close();
  }

  /* This is the main function for the consumer thread.
   * It will wait at a semaphore for pending work. When there is
   * work on one or more queues, it will iterate over the queues
   * and process the pending profiler objects, updating the profiles
   * as it goes. */
  void profiler_listener::process_profiles(void)
  {
    profiler * p;
    unsigned int i;
    // Main loop. Stay in this loop unless "done".
    while (!done) {
      queue_signal.wait();
      for (i = 0 ; i < profiler_queues.size(); i++) {
        if (profiler_queues[i]) {
          while (profiler_queues[i]->pop(p)) {
            process_profile(p, i);
          }
        }
      }
    }

    // We might be done, but check to make sure the queues are empty
    for (i = 0 ; i < profiler_queues.size(); i++) {
      if (profiler_queues[i]) {
        while (profiler_queues[i]->pop(p)) {
          process_profile(p, i);
        }
      }
    }

    // stop the main timer, and process that profile
    main_timer->stop();
    process_profile(main_timer, my_tid);

    // output to screen?
    if (apex_options::use_screen_output() && node_id == 0)
    {
      finalize_profiles();
    }

    // output to TAU profiles?
    if (apex_options::use_profile_output())
    {
      // the number of thread_name_maps tells us how many threads there are to process
      for (i = 0 ; i < thread_name_maps.size(); i++) {
        write_profile(i);
      }
    }

    // cleanup.
    delete_profiles();
  }


#if APEX_HAVE_PAPI
__thread int EventSet = PAPI_NULL;
#define PAPI_ERROR_CHECK(name) \
if (rc != 0) cout << "name: " << rc << ": " << PAPI_strerror(rc) << endl;

  void initialize_PAPI(bool first_time) {
      int rc = 0;
      if (first_time) {
        PAPI_library_init( PAPI_VER_CURRENT );
        PAPI_thread_init( thread_instance::get_id );
        rc = PAPI_set_domain(PAPI_DOM_ALL);
        PAPI_ERROR_CHECK(PAPI_set_domain);
      }
      rc = PAPI_create_eventset( &EventSet );
      PAPI_ERROR_CHECK(PAPI_create_eventset);
      rc = PAPI_assign_eventset_component (EventSet, 0);
      PAPI_ERROR_CHECK(PAPI_assign_eventset_component);
      rc = PAPI_set_granularity(PAPI_GRN_THR);
      PAPI_ERROR_CHECK(PAPI_set_granularity);
      rc = PAPI_add_event( EventSet, PAPI_TOT_CYC);
      PAPI_ERROR_CHECK(PAPI_add_event);
      rc = PAPI_add_event( EventSet, PAPI_TOT_INS);
      PAPI_ERROR_CHECK(PAPI_add_event);
      /*
      rc = PAPI_add_event( EventSet, PAPI_FP_OPS);
      PAPI_ERROR_CHECK(PAPI_add_event);
      rc = PAPI_add_event( EventSet, PAPI_FP_INS);
      PAPI_ERROR_CHECK(PAPI_add_event);
      */
      rc = PAPI_add_event( EventSet, PAPI_L2_TCM);
      PAPI_ERROR_CHECK(PAPI_add_event);
      rc = PAPI_start( EventSet );
      PAPI_ERROR_CHECK(PAPI_start);
  }

#endif

  /* When APEX gets a STARTUP event, do some initialization. */
  void profiler_listener::on_startup(startup_event_data &data) {
    if (!_terminate) {
      // Create a profiler queue for this main thread
      profiler_queues[0] = new boost::lockfree::spsc_queue<profiler*>(MAX_QUEUE_SIZE);
      thread_address_maps[0] = new map<void*, profile*>();
      thread_name_maps[0] = new map<string, profile*>();
      // Start the consumer thread, to process profiler objects.
      consumer_thread = new boost::thread(process_profiles);

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
      main_timer = new profiler(new string(APEX_MAIN));
    }
  }

  /* On the shutdown event, notify the consumer thread that we are done
   * and set the "terminate" flag. */
  void profiler_listener::on_shutdown(shutdown_event_data &data) {
    if (!_terminate) {
      node_id = data.node_id;
      _terminate = true;
      done = true;
      queue_signal.post();
      consumer_thread->join();
#if APEX_HAVE_PAPI
      int rc = 0;
      int i = 0;
      long long values[8] {0L};
      //cout << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << endl;
      for (i = 0 ; i < thread_instance::get_num_threads() ; i++) {
        rc = PAPI_accum( event_sets[i], values );
        PAPI_ERROR_CHECK(PAPI_stop);
        //cout << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << endl;
      }
      if (apex_options::use_screen_output() && node_id == 0) {
        cout << endl << "TOTAL COUNTERS for " << thread_instance::get_num_threads() << " threads:" << endl;
        cout << "Cycles: " << values[0] ;
        cout << ", Instructions: " << values[1] ;
        //cout << ", FPOPS: " << values[2] ;
        //cout << ", FPINS: " << values[3] ;
        cout << ", L2TCM: " << values[2] ;

        cout << endl << "IPC: " << (double)(values[1])/(double)(values[0]) ;
        //cout << endl << "FLOP%INS: " << (double)(values[2])/(double)(values[1]) ;
        //cout << endl << "FLINS%INS: " << (double)(values[3])/(double)(values[1]) ;
        cout << endl << "INS/L2TCM: " << (double)(values[1])/(double)(values[2]) ;
        cout << endl;
      }
#endif
    }
  }

  /* When a new node is created */
  void profiler_listener::on_new_node(node_event_data &data) {
    if (!_terminate) {
    }
  }

  /* When a new thread is registered, expand all of our storage as necessary
   * to handle the new thread */
  void profiler_listener::on_new_thread(new_thread_event_data &data) {
    if (!_terminate) {
      _mtx.lock();
      // resize the vector
      my_tid = (unsigned int)thread_instance::get_id();
      if (my_tid >= profiler_queues.size()) {
        profiler_queues.resize(my_tid + 1);
      }
      if (my_tid >= thread_address_maps.size()) {
        thread_address_maps.resize(my_tid + 1);
      }
      if (my_tid >= thread_name_maps.size()) {
        thread_name_maps.resize(my_tid + 1);
      }
      unsigned int i = 0;
      // allocate the queue(s)
      for (i = 0; i < my_tid+1 ; i++) {
        if (profiler_queues[i] == NULL) {
          boost::lockfree::spsc_queue<profiler*>* tmp = 
            new boost::lockfree::spsc_queue<profiler*>(MAX_QUEUE_SIZE);
          profiler_queues[i] = tmp;
        }
	if (thread_address_maps[i] == NULL) {
          thread_address_maps[i] = new map<void*, profile*>();
	}
	if (thread_name_maps[i] == NULL) {
          thread_name_maps[i] = new map<string, profile*>();
	}
      }
#if APEX_HAVE_PAPI
      initialize_PAPI(false);
      if (my_tid >= event_sets.size()) {
        event_sets.resize(my_tid + 1);
      }
      event_sets[my_tid] = EventSet;
#endif
      _mtx.unlock();
    }
  }

  /* When a start event happens, create a profiler object. Unless this 
   * named event is throttled, in which case do nothing, as quickly as possible */
  void profiler_listener::on_start(apex_function_address function_address, string *timer_name) {
    if (!_terminate) {
      if (timer_name != NULL) {
#if defined(APEX_THROTTLE)
        // if this timer is throttled, return without doing anything
        unordered_set<void*>::const_iterator it = throttled_names.find(*timer_name);
        if (it != throttled_names.end()) {
          thread_instance::instance().current_timer = NULL;
          return;
        }
#endif
        // start the profiler object, which starts our timers
        thread_instance::instance().current_timer = new profiler(timer_name);
      } else {
#if defined(APEX_THROTTLE)
        // if this timer is throttled, return without doing anything
        unordered_set<void*>::const_iterator it = throttled_addresses.find(function_address);
        if (it != throttled_addresses.end()) {
          thread_instance::instance().current_timer = NULL;
          return;
        }
#endif
        // start the profiler object, which starts our timers
        thread_instance::instance().current_timer = new profiler(function_address);
      }
#if APEX_HAVE_PAPI
      long long * values = thread_instance::instance().current_timer->papi_start_values;
      int rc = 0;
      rc = PAPI_read( EventSet, values );
      PAPI_ERROR_CHECK(PAPI_read);
#endif
    }
  }

  /* Stop the timer, if applicable, and queue the profiler object */
  void profiler_listener::on_stop(profiler * p) {
    if (!_terminate) {
      if (p) {
        p->stop();
#if APEX_HAVE_PAPI
        long long * values = p->papi_stop_values;
        int rc = 0;
        rc = PAPI_read( EventSet, values );
        PAPI_ERROR_CHECK(PAPI_read);
	/*
        if (p->have_name)
          cout << p->timer_name;
	else
          cout << *(lookup_address((uintptr_t)p->action_address));
	cout << endl;
        cout << "Cycles: " << values[0] ;
        cout << ", Instructions: " << values[1] ;
        cout << ", FPOPS: " << values[2] ;
        cout << ", FPINS: " << values[3] ;
        cout << endl << "IPC: " << (double)(values[1])/(double)(values[0]) ;
        cout << endl << "FLOP%INS: " << (double)(values[2])/(double)(values[1]) ;
        cout << endl << "FLINS%INS: " << (double)(values[3])/(double)(values[1]) ;
        cout << endl;
	*/
#endif
        profiler_queues[my_tid]->push(p);
        queue_signal.post();
      }
    }
  }

  /* This is just like starting a timer, but don't increment the number of calls
   * value. That is because we are restarting an existing timer. */
  void profiler_listener::on_resume(profiler *p) {
    if (!_terminate) {
      if (p->have_name) {
        thread_instance::instance().current_timer = new profiler(p->timer_name, true);
      } else {
        thread_instance::instance().current_timer = new profiler(p->action_address, true);
      }
    }
  }

  /* When a sample value is processed, save it as a profiler object, and queue it. */
  void profiler_listener::on_sample_value(sample_value_event_data &data) {
    if (!_terminate) {
      profiler * p = new profiler(new string(*data.counter_name), data.counter_value);
      profiler_queues[my_tid]->push(p);
      queue_signal.post();
    }
  }

  /* For periodic stuff. Do something? */
  void profiler_listener::on_periodic(periodic_event_data &data) {
    if (!_terminate) {
    }
  }

#if APEX_HAVE_BFD

  /*
   *-----------------------------------------------------------------------------
   * Simple hash table to map function addresses to region names/identifier
   *-----------------------------------------------------------------------------
   */

  struct OmpHashNode
  {
    OmpHashNode() { }

    ApexBfdInfo info;        ///< Filename, line number, etc.
    string * location;
  };

  /* destructor helper */
  extern void Apex_delete_hash_table(void);

  /* Define the table of addresses to names */
  struct OmpHashTable : public std::map<uintptr_t, OmpHashNode*>
  {
    OmpHashTable() { }
    virtual ~OmpHashTable() {
      Apex_delete_hash_table();
    }
  };

  /* Static constructor. We only need one. */
  static OmpHashTable & OmpTheHashTable()
  {
    static OmpHashTable htab;
    return htab;
  }

  /* Static BFD unit handle generator. We only need one. */
  static apex_bfd_handle_t & OmpTheBfdUnitHandle()
  {
    static apex_bfd_handle_t OmpbfdUnitHandle = APEX_BFD_NULL_HANDLE;
    if (OmpbfdUnitHandle == APEX_BFD_NULL_HANDLE) {
      if (OmpbfdUnitHandle == APEX_BFD_NULL_HANDLE) {
        OmpbfdUnitHandle = Apex_bfd_registerUnit();
      }
    }
    return OmpbfdUnitHandle;
  }

  /* Delete the hash table. */
  void Apex_delete_hash_table(void) {
    // clear the hash map to eliminate memory leaks
    OmpHashTable & mytab = OmpTheHashTable();
    for ( std::map<uintptr_t, OmpHashNode*>::iterator it = mytab.begin(); it != mytab.end(); ++it ) {
      OmpHashNode * node = it->second;
      if (node->location) {
        delete (node->location);
      }
      delete node;
    }
    mytab.clear();
    Apex_delete_bfd_units();
  }

  /* Map a function address to a name and/or source location */
  string * lookup_address(uintptr_t ip) {
    stringstream location;
    apex_bfd_handle_t & OmpbfdUnitHandle = OmpTheBfdUnitHandle();
    OmpHashNode * node = OmpTheHashTable()[ip];
    if (!node) {
      node = new OmpHashNode;
      Apex_bfd_resolveBfdInfo(OmpbfdUnitHandle, ip, node->info);
      location << node->info.funcname << " [{" << node->info.filename << "} {" << node->info.lineno << ",0}]";
      node->location = new string(location.str());
      OmpTheHashTable()[ip] = node;
    }
    return node->location;
  }

#endif

}
