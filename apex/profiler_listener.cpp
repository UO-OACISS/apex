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

//#define MAX_QUEUE_SIZE 1024*1024
#define MAX_QUEUE_SIZE 128

using namespace std;

namespace apex {

std::vector<boost::lockfree::spsc_queue<profiler*>* > profiler_queues(8);
boost::thread * consumer_thread;
boost::atomic<bool> done (false);
semaphore queue_signal;
static map<string, profile*> name_map;
static map<void*, profile*> address_map;
#if defined(APEX_THROTTLE)
static unordered_set<void*> throttled_addresses;
#endif

profiler * profiler_listener::main_timer(NULL);
int profiler_listener::node_id(0);

profile * profiler_listener::get_profile(apex_function_address address) {
   map<void*, profile*>::iterator it = address_map.find((void*)address);
   if (it != address_map.end()) {
     return (*it).second;
   }
   return NULL;
}

profile * profiler_listener::get_profile(string &timer_name) {
   map<string, profile*>::iterator it = name_map.find(timer_name);
   if (it != name_map.end()) {
     return (*it).second;
   }
   return NULL;
}

#if APEX_HAVE_BFD
extern string * lookup_address(uintptr_t ip);
#endif

inline void profiler_listener::process_profile(profiler * p)
{
    profile * theprofile;
    if (p->have_name) {
      map<string, profile*>::iterator it = name_map.find(*(p->timer_name));
      if (it != name_map.end()) {
        // A profile for this name already exists.
        theprofile = (*it).second;
        theprofile->increment(p->elapsed());
      } else {
        // Create a new profile for this name.
        theprofile = new profile(p->elapsed(), p->is_counter ? COUNTER : TIMER);
        name_map[*(p->timer_name)] = theprofile;
	// done with the name now
	delete(p->timer_name);
      }
    } else {
      map<void*, profile*>::const_iterator it2 = address_map.find(p->action_address);
      if (it2 != address_map.end()) {
        theprofile = (*it2).second;
        theprofile->increment(p->elapsed());
#if defined(APEX_THROTTLE)
        if (theprofile->get_calls() > APEX_THROTTLE_CALLS && 
            theprofile->get_mean() < APEX_THROTTLE_PERCALL) { 
          unordered_set<void*>::iterator it = throttled_addresses.find(p->action_address);
          if (it == throttled_addresses.end()) { 
	    throttled_addresses.insert(p->action_address);
	    cout << "APEX Throttled " << *(lookup_address((uintptr_t)p->action_address)) << endl;
	  }
	}
#endif
      } else {
        theprofile = new profile(p->elapsed());
        address_map[p->action_address] = theprofile;
      }
    }
    delete(p);
}

void profiler_listener::delete_profiles(void) {
#if 1
    map<void*, profile*>::const_iterator it;
    for(it = address_map.begin(); it != address_map.end(); it++) {
        delete it->second;
	//address_map.erase(it->first);
    }
    map<string, profile*>::const_iterator it2;
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
        delete it2->second;
	//name_map.erase(it2->first);
    }
#endif
    address_map.clear();
    name_map.clear();
    unsigned int i = 0;
    for (i = 0 ; i < profiler_queues.size(); i++) {
	if (profiler_queues[i]) {
            delete (profiler_queues[i]);
	}
    }
    profiler_queues.clear();
}

void profiler_listener::finalize_profiles(void) {
    map<void*, profile*>::const_iterator it;
    for(it = address_map.begin(); it != address_map.end(); it++) {
      profile * p = it->second;
      void * function_address = it->first;
#if defined(APEX_THROTTLE)
      unordered_set<void*>::const_iterator it = throttled_addresses.find(function_address);
      if (it != throttled_addresses.end()) { continue; }
#endif
#if APEX_HAVE_BFD
      string * tmp = lookup_address((uintptr_t)function_address);
      cout << *tmp << ": " ;
#else
      cout << function_address << ": " ;
#endif
      cout << p->get_calls() << ", " ;
      cout << p->get_minimum() << ", " ;
      cout << p->get_mean() << ", " ;
      cout << p->get_maximum() << ", " ;
      cout << p->get_accumulated() << ", " ;
      cout << p->get_stddev() << endl;
    }
    map<string, profile*>::const_iterator it2;
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      string action_name = it2->first;
      cout << action_name << ": " ;
      cout << p->get_calls() << ", " ;
      cout << p->get_minimum() << ", " ;
      cout << p->get_mean() << ", " ;
      cout << p->get_maximum() << ", " ;
      cout << p->get_accumulated() << ", " ;
      cout << p->get_stddev() << endl;
    }
}

void format_line(ofstream &myfile, profile * p) {
    myfile << p->get_calls() << " ";
    myfile << 0 << " ";
    myfile << (p->get_accumulated() * 1000000.0) << " ";
    myfile << (p->get_accumulated() * 1000000.0) << " ";
    myfile << 0 << " ";
    myfile << "GROUP=\"TAU_USER\" ";
    myfile << endl;
}

void format_counter_line(ofstream &myfile, profile * p) {
    myfile << p->get_calls() << " ";       // numevents
    myfile << p->get_maximum() << " ";     // max
    myfile << p->get_minimum() << " ";     // min
    myfile << p->get_mean() << " ";        // mean
    myfile << p->get_sum_squares() << " "; 
    myfile << endl;
}

void profiler_listener::write_profile(void) {
    ofstream myfile;
    stringstream datname;
    datname << "profile." << node_id << ".0.0";
    myfile.open(datname.str().c_str());
    int counter_events = 0;

    // Determine number of counter events, as these need to be
    // excluded from the number of normal timers
    map<string, profile*>::const_iterator it2;
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == COUNTER) {
        counter_events++;
      }
    }
    int function_count = address_map.size() + (name_map.size() - counter_events);

    // Print the normal timers to the profile file
    // 1504 templated_functions_MULTI_TIME
    myfile << function_count << " templated_functions_MULTI_TIME" << endl;
    // # Name Calls Subrs Excl Incl ProfileCalls #
    myfile << "# Name Calls Subrs Excl Incl ProfileCalls #" << endl;
    thread_instance ti = thread_instance::instance();

    // Iterate over the profiles which are associated to a function
    // by address. All of these are regular timers.
    map<void*, profile*>::const_iterator it;
    for(it = address_map.begin(); it != address_map.end(); it++) {
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
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == TIMER) {
        string action_name = it2->first;
        myfile << "\"" << action_name << "\" ";
        format_line (myfile, p);
      }
    }

    // 0 aggregates
    myfile << "0 aggregates" << endl;

    // Now process the counters, if there are any.
    if(counter_events > 0) {
      myfile << counter_events << " userevents" << endl;
      myfile << "# eventname numevents max min mean sumsqr" << endl;
      for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
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

void profiler_listener::process_profiles(void)
{
    profiler * p;
    unsigned int i;
    while (!done) {
        queue_signal.wait();
	for (i = 0 ; i < profiler_queues.size(); i++) {
	    if (profiler_queues[i]) {
                while (profiler_queues[i]->pop(p)) {
                    process_profile(p);
                }
	    }
	}
    }

    for (i = 0 ; i < profiler_queues.size(); i++) {
	if (profiler_queues[i]) {
            while (profiler_queues[i]->pop(p)) {
                process_profile(p);
            }
        }
    }
    main_timer->stop();
    process_profile(main_timer);

    if (apex_options::use_screen_output())
    {
      finalize_profiles();
    }
    if (apex_options::use_profile_output())
    {
      write_profile();
    }
    delete_profiles();
}

void profiler_listener::on_startup(startup_event_data &data) {
  if (!_terminate) {
      //cout << "STARTUP" << endl;
      // one for the main thread
      profiler_queues[0] = new boost::lockfree::spsc_queue<profiler*>(MAX_QUEUE_SIZE);
      // one for the worker thread
      //profiler_queues[1] = new boost::lockfree::spsc_queue<profiler*>(MAX_QUEUE_SIZE);
      consumer_thread = new boost::thread(process_profiles);
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

      main_timer = new profiler(new string("APEX MAIN"));
  }
}

void profiler_listener::on_shutdown(shutdown_event_data &data) {
  if (!_terminate) {
      //cout << "SHUTDOWN" << endl;
      node_id = data.node_id;
      _terminate = true;
      done = true;
      queue_signal.post();
      consumer_thread->join();
  }
}

void profiler_listener::on_new_node(node_event_data &data) {
  if (!_terminate) {
      //cout << "NEW NODE" << endl;
  }
}

void profiler_listener::on_new_thread(new_thread_event_data &data) {
  if (!_terminate) {
      //cout << "NEW THREAD" << endl;
      unsigned int me = (unsigned int)thread_instance::get_id();
      if (me >= profiler_queues.size()) {
	      profiler_queues.resize(me + 1);
      }
	  unsigned int i = 0;
	  for (i = 0; i < me+1 ; i++) {
	    if (profiler_queues[i] == NULL) {
          boost::lockfree::spsc_queue<profiler*>* tmp = new boost::lockfree::spsc_queue<profiler*>(MAX_QUEUE_SIZE);
          profiler_queues[i] = tmp;
		}
	  }
  }
}

void profiler_listener::on_start(apex_function_address function_address, string *timer_name) {
  if (!_terminate) {
      if (timer_name != NULL) {
        thread_instance::instance().current_timer = new profiler(timer_name);
      } else {
#if defined(APEX_THROTTLE)
        unordered_set<void*>::const_iterator it = throttled_addresses.find(function_address);
        if (it != throttled_addresses.end()) {
          thread_instance::instance().current_timer = NULL;
	  return;
	}
#endif
        thread_instance::instance().current_timer = new profiler(function_address);
      }
  }
}

void profiler_listener::on_stop(profiler * p) {
  //static __thread int counter = 0; // only do 1/10 of the timers
  if (!_terminate) {
      if (p) {
          p->stop();
          int me = thread_instance::get_id();
          profiler_queues[me]->push(p);
          queue_signal.post();
      }
  }
}

void profiler_listener::on_resume(profiler *p) {
  if (!_terminate) {
      if (p->have_name) {
        thread_instance::instance().current_timer = new profiler(p->timer_name);
      } else {
        thread_instance::instance().current_timer = new profiler(p->action_address);
      }
  }
}

void profiler_listener::on_sample_value(sample_value_event_data &data) {
  if (!_terminate) {
      profiler * p = new profiler(new string(*data.counter_name), data.counter_value);
      int me = thread_instance::get_id();
      profiler_queues[me]->push(p);
      queue_signal.post();
  }
}

void profiler_listener::on_periodic(periodic_event_data &data) {
  if (!_terminate) {
      //cout << "PERIODIC" << endl;
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

extern void Apex_delete_hash_table(void);

struct OmpHashTable : public std::map<uintptr_t, OmpHashNode*>
{
  OmpHashTable() { }
  virtual ~OmpHashTable() {
    Apex_delete_hash_table();
  }
};

static OmpHashTable & OmpTheHashTable()
{
  static OmpHashTable htab;
  return htab;
}

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
