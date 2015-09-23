//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "profiler_listener.hpp"
#include "profiler.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include "apex_options.hpp"
#include "profiler.hpp"
#include "profile.hpp"
#include "apex.hpp"

#include <boost/thread/thread.hpp>
#include <boost/atomic.hpp>
#include <atomic>
#include <boost/range/adaptor/map.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/assign.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#include <unistd.h>
#include <sched.h>
#endif
#include <cstdio>
#include <vector>
#include <string>
#include <boost/regex.hpp>

#if defined(APEX_THROTTLE)
#define APEX_THROTTLE_CALLS 1000
#define APEX_THROTTLE_PERCALL 0.00001 // 10 microseconds.
#endif
#include <unordered_set>

#if APEX_HAVE_BFD
#include "address_resolution.hpp"
#endif

#if APEX_HAVE_PAPI
#include "papi.h"
#endif

#ifdef APEX_HAVE_HPX3
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos/local/composable_guard.hpp>
#endif

#define APEX_MAIN "APEX MAIN"

#ifdef APEX_HAVE_TAU
#define PROFILING_ON
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif

#include <future>
#include <thread>

using namespace std;
using namespace apex;

APEX_NATIVE_TLS unsigned int my_tid = 0; // the current thread's TID in APEX

namespace apex {

  /* THis is a special profiler, indicating that the timer requested is
     throttled, and shouldn't be processed. */
  profiler* profiler::disabled_profiler = new profiler();

  /* Return the requested profile object to the user.
   * Return nullptr if doesn't exist. */
  profile * profiler_listener::get_profile(apex_function_address address) {
    map<apex_function_address, profile*>::const_iterator it = address_map.find(address);
#ifdef APEX_HAVE_TAU
	profile * p = nullptr;
    // get the most recent data
	const char * list[1];
	double **counterExclusiveValues;
	double **counterInclusiveValues;
	int *numOfCalls;
	int *numOfSubRoutines;
    const char ** counterNames;
	int numOfCounters;
	list[0] = thread_instance::instance().map_addr_to_name(address).c_str();
	Tau_get_function_values(list, 1, &counterExclusiveValues, &counterInclusiveValues, 
	      &numOfCalls, &numOfSubRoutines, &counterNames, &numOfCounters);
    if (it != address_map.end()) {
	  p = (*it).second;
    }
	if (p == nullptr) {
	  p = new profile(counterInclusiveValues[0][0], false);
	}
    p->set_accumulated(counterInclusiveValues[0][0]);
    p->set_calls(numOfCalls[0]);
    p->set_minimum(counterInclusiveValues[0][0]);
    p->set_maximum(counterInclusiveValues[0][0]);
	return p;
#else
    if (it != address_map.end()) {
      return (*it).second;
    }
    return nullptr;
#endif
  }

  /* Return the requested profile object to the user.
   * Return nullptr if doesn't exist. */
  profile * profiler_listener::get_profile(const string &timer_name) {
    map<string, profile*>::const_iterator it = name_map.find(timer_name);
#ifdef APEX_HAVE_TAU
	profile * p = nullptr;
    if (it != name_map.end()) {
	  p = (*it).second;
    }
	if (p == nullptr) {
	  p = new profile(0.0, false);
	}
    // get the most recent data
	const char * list[1];
	double **counterExclusiveValues;
	double **counterInclusiveValues;
	int *numOfCalls;
	int *numOfSubRoutines;
    const char **counterNames;
	int numOfCounters;
	list[0] = timer_name.c_str();
	Tau_get_function_values(list, 1, &counterExclusiveValues, &counterInclusiveValues, 
	      &numOfCalls, &numOfSubRoutines, &counterNames, &numOfCounters);
    if (numOfCalls[0] == 0) { // doesn't exist, must be a counter
	  double *max;
	  double *min;
	  double *mean;
	  double *sumSqr;
	  Tau_get_event_vals(list, 1, &numOfCalls, &max, &min, &mean, &sumSqr);
      p->set_calls(numOfCalls[0]);
      p->set_accumulated(numOfCalls[0] * mean[0]);
      p->set_minimum(min[0]);
      p->set_maximum(max[0]);
	} else {
      p->set_calls(numOfCalls[0]);
      p->set_accumulated(counterInclusiveValues[0][0]);
      p->set_minimum(counterInclusiveValues[0][0]);
      p->set_maximum(counterInclusiveValues[0][0]);
	}
	return p;
#else
    if (it != name_map.end()) {
      return (*it).second;
    }
    return nullptr;
#endif
  }

  /* Return a vector of all name-based profiles */
  std::vector<std::string> profiler_listener::get_available_profiles() {
    std::vector<std::string> names;
    boost::copy(name_map | boost::adaptors::map_keys, std::back_inserter(names));
    return names;
  }

  void profiler_listener::reset_all(void) {
    for(auto &it : name_map) {
        it.second->reset();
    }
    for(auto &it : address_map) {
        it.second->reset();
    }
    if (apex_options::use_profile_output() > 1) {
        for(auto &it1 : *(thread_name_maps[my_tid])) {
            it1.second->reset();
        }
        for(auto &it1 : *(thread_address_maps[my_tid])) {
            it1.second->reset();
        }
    }
  }

  /* After the consumer thread pulls a profiler off of the queue,
   * process it by updating its profile object in the map of profiles. */
  // TODO The name-based timer and address-based timer paths through
  // the code involve a lot of duplication -- this should be refactored
  // to remove the duplication so it's easier to maintain.
  inline unsigned int profiler_listener::process_profile(profiler* p, unsigned int tid)
  {
    if(p == nullptr) return 0;
    profile * theprofile;
    if(p->is_reset == reset_type::ALL) {
        reset_all();
        delete p;
        return 0;
    }
    // Look for the profile object by name, if applicable
    if (p->have_name) {
      map<string, profile*>::const_iterator it = name_map.find(*(p->timer_name));
      if (it != name_map.end()) {
        // A profile for this name already exists.
        theprofile = (*it).second;
        if(p->is_reset == reset_type::CURRENT) {
            theprofile->reset();
        } else {
            theprofile->increment(p->elapsed(), p->is_resume);
        }
#if defined(APEX_THROTTLE)
        // Is this a lightweight task? If so, we shouldn't measure it any more,
        // in order to reduce overhead.
        if (theprofile->get_calls() > APEX_THROTTLE_CALLS &&
            theprofile->get_mean() < APEX_THROTTLE_PERCALL) {
          unordered_set<string>::const_iterator it2 = throttled_names.find(*(p->timer_name));
          if (it2 == throttled_names.end()) {
            throttled_names.insert(*(p->timer_name));
            cout << "APEX Throttled " << p->timer_name << endl; fflush(stdout);
          }
        }
#endif
      } else {
        // Create a new profile for this name.
        theprofile = new profile(p->is_reset == reset_type::CURRENT ? 0.0 : p->elapsed(), p->is_resume, p->is_counter ? APEX_COUNTER : APEX_TIMER);
        name_map[*(p->timer_name)] = theprofile;
#ifdef APEX_HAVE_HPX3
#ifdef APEX_REGISTER_HPX3_COUNTERS
        if(!_done) {
            if(get_hpx_runtime_ptr() != nullptr) {
                std::string timer_name(*(p->timer_name));
                //Don't register timers containing "/"
                if(timer_name.find("/") == std::string::npos) {
                    hpx::performance_counters::install_counter_type(
                    std::string("/apex/") + timer_name,
                    [p](bool r)->boost::int64_t{
                        boost::int64_t value(p->elapsed() * 100000);
                        delete p;
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
      if (apex_options::use_profile_output() > 1) {
        // now do thread-specific measurement.
        map<string, profile*>* the_map = thread_name_maps[tid];
        it = the_map->find(*(p->timer_name));
        if (it != the_map->end()) {
            // A profile for this name already exists.
            theprofile = (*it).second;
            if(p->is_reset == reset_type::CURRENT) {
                theprofile->reset();
            } else {
                theprofile->increment(p->elapsed(), p->is_resume);
            }
        } else {
            // Create a new profile for this name.
            theprofile = new profile(p->is_reset == reset_type::CURRENT ? 0.0 : p->elapsed(), p->is_resume, p->is_counter ? APEX_COUNTER : APEX_TIMER);
            (*the_map)[*(p->timer_name)] = theprofile;
        }
      }
    } else { // address rather than name
      map<apex_function_address, profile*>::const_iterator it2 = address_map.find(p->action_address);
      if (it2 != address_map.end()) {
        // A profile for this name already exists.
        theprofile = (*it2).second;
        if(p->is_reset == reset_type::CURRENT) {
            theprofile->reset();
        } else {
            theprofile->increment(p->elapsed(), p->is_resume);
        }
#if defined(APEX_THROTTLE)
        // Is this a lightweight task? If so, we shouldn't measure it any more,
        // in order to reduce overhead.
        if (theprofile->get_calls() > APEX_THROTTLE_CALLS &&
            theprofile->get_mean() < APEX_THROTTLE_PERCALL) {
          unordered_set<apex_function_address>::const_iterator it4 = throttled_addresses.find(p->action_address);
          if (it4 == throttled_addresses.end()) {
            throttled_addresses.insert(p->action_address);
#if defined(HAVE_BFD)
            cout << "APEX Throttled " << *(lookup_address((uintptr_t)p->action_address, true)) << endl; fflush(stdout);
#else
            cout << "APEX Throttled " << p->action_address << endl; fflush(stdout);
#endif
          }
        }
#endif
      } else {
        // Create a new profile for this address.
        theprofile = new profile(p->is_reset == reset_type::CURRENT ? 0.0 : p->elapsed(), p->is_resume);
        address_map[p->action_address] = theprofile;
      }
      if (apex_options::use_profile_output() > 1) {
        // now do thread-specific measurement
        map<apex_function_address, profile*>* the_map = thread_address_maps[tid];
        it2 = the_map->find(p->action_address);
        if (it2 != the_map->end()) {
            // A profile for this name already exists.
            theprofile = (*it2).second;
            if(p->is_reset == reset_type::CURRENT) {
                theprofile->reset();
            } else {
                theprofile->increment(p->elapsed(), p->is_resume);
            }
        } else {
            // Create a new profile for this address.
            theprofile = new profile(p->is_reset == reset_type::CURRENT ? 0.0 : p->elapsed(), p->is_resume);
            (*the_map)[p->action_address] = theprofile;
        }
      }
    }
    delete p;
    return 1;
  }

  inline unsigned int profiler_listener::process_dependency(task_dependency* td)
  {
      unordered_map<task_identifier, unordered_map<task_identifier, int>* >::const_iterator it = task_dependencies.find(*td->parent);
      unordered_map<task_identifier, int> * depend;
      // if this is a new dependency for this parent?
      if (it == task_dependencies.end()) {
          depend = new unordered_map<task_identifier, int>();
          depend->emplace(*td->child, 1);
          task_dependencies.emplace(*td->parent, depend);
      // otherwise, see if this parent has seen this child
      } else {
          depend = it->second;
          unordered_map<task_identifier, int>::const_iterator it2 = depend->find(*td->child);
          // first time for this child
          if (it2 == depend->end()) {
              depend->emplace(*td->child, 1);
          // not the first time for this child
          } else {
              int tmp = it2->second;
              (*depend)[*td->child] = tmp + 1;
          }
      }
      delete(td);
      return 1;
  }

  /* Cleaning up memory. Not really necessary, because it only gets
   * called at shutdown. But a good idea to do regardless. */
  void profiler_listener::delete_profiles(void) {
    // iterate over the map and free the objects in the map
    map<apex_function_address, profile*>::const_iterator it;
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

    if (apex_options::use_profile_output() > 1) {
        unsigned int i;
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
  }

#define PAD_WITH_SPACES boost::format("%9i")
#define FORMAT_SCIENTIFIC boost::format("%1.3e")

  /* At program termination, write the measurements to the screen. */
  void profiler_listener::finalize_profiles(void) {
    // iterate over the profiles in the address map
    map<apex_function_address, profile*>::const_iterator it;
    cout << "Action                         :   #calls  |  minimum  |    mean   |  maximum  |   total   |  stddev  " << endl;
    cout << "------------------------------------------------------------------------------------------------------" << endl;
    for(it = address_map.begin(); it != address_map.end(); it++) {
      profile * p = it->second;
      apex_function_address function_address = it->first;
#if APEX_HAVE_BFD
      // translate the address to a name
      string * tmp = lookup_address((uintptr_t)function_address, true);
      string shorter(*tmp);
      // to keep formatting pretty, trim any long timer names
      if (shorter.size() > 30) {
        shorter.resize(27);
        shorter.resize(30, '.');
      }
      //cout << "\"" << shorter << "\", " ;
      cout << boost::format("%30s") % shorter << " : ";
#else
      //cout << "\"" << function_address << "\", " ;
      cout << boost::format("%30p") % function_address << " : " ;
#endif
#if defined(APEX_THROTTLE)
      // if this profile was throttled, don't output the measurements.
      // they are limited and bogus, anyway.
      unordered_set<apex_function_address>::const_iterator it3 = throttled_addresses.find(function_address);
      if (it3 != throttled_addresses.end()) { 
        cout << "THROTTLED (high frequency, short duration)" << endl;
        continue; 
      }
#endif
      if (p->get_calls() < 999999) {
          cout << PAD_WITH_SPACES % p->get_calls() << "   " ;
      } else {
          cout << FORMAT_SCIENTIFIC % p->get_calls() << "   " ;
      }
      cout << FORMAT_SCIENTIFIC % p->get_minimum() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_mean() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_maximum() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_accumulated() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_stddev() << endl;
    }
    map<string, profile*>::const_iterator it2;
    // iterate over the profiles in the name map
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      string action_name = it2->first;
#if APEX_HAVE_BFD
      boost::regex rx (".*UNRESOLVED ADDR (.*)");
      if (boost::regex_match (action_name,rx)) {
        const boost::regex separator(" ADDR ");
        boost::sregex_token_iterator token(action_name.begin(), action_name.end(), separator, -1);
        *token++; // ignore
        string addr_str = *token++;
    void* addr_addr;
    sscanf(addr_str.c_str(), "%p", &addr_addr);
        string * tmp = lookup_address((uintptr_t)addr_addr, true);
        boost::regex old_address("UNRESOLVED ADDR " + addr_str);
    action_name = boost::regex_replace(action_name, old_address, *tmp);
      }
#endif
      string shorter(action_name);
      // to keep formatting pretty, trim any long timer names
      if (shorter.size() > 30) {
        shorter.resize(27);
        shorter.resize(30, '.');
      }
      //cout << "\"" << shorter << "\", " ;
      cout << boost::format("%30s") % shorter << " : ";
#if defined(APEX_THROTTLE)
      // if this profile was throttled, don't output the measurements.
      // they are limited and bogus, anyway.
      /*
      unordered_set<string>::const_iterator it4 = throttled_names.find(action_name);
      if (it4!= throttled_names.end()) { 
        cout << "THROTTLED (high frequency, short duration)" << endl;
        continue; 
      }
      */
#endif
      if(p->get_calls() < 1) {
        p->get_profile()->calls = 1;
      }
      if (p->get_calls() < 999999) {
          cout << PAD_WITH_SPACES % p->get_calls() << "   " ;
      } else {
          cout << FORMAT_SCIENTIFIC % p->get_calls() << "   " ;
      }
      cout << FORMAT_SCIENTIFIC % p->get_minimum() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_mean() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_maximum() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_accumulated() << "   " ;
      cout << FORMAT_SCIENTIFIC % p->get_stddev() << endl;
    }
  }

  void fix_name(string& in_name) {
#if defined(HAVE_BFD)                                                            
        boost::regex rx (".*UNRESOLVED ADDR (.*)");
        if (boost::regex_match (in_name,rx)) {
          const boost::regex separator(" ADDR ");
          boost::sregex_token_iterator token(in_name.begin(), in_name.end(), separator, -1);
          *token++; // ignore
          string addr_str = *token++;
          void* addr_addr;
          sscanf(addr_str.c_str(), "%p", &addr_addr);
          string tmp = lookup_address((uintptr_t)addr_addr, false);
          boost::regex old_address("UNRESOLVED ADDR " + addr_str);
          in_name = boost::regex_replace(in_name, old_address, tmp);
        }
#endif
  }

  void profiler_listener::write_taskgraph(void) {
    ofstream myfile;
    stringstream dotname;
    dotname << "taskgraph." << node_id << ".dot";
    myfile.open(dotname.str().c_str());

    myfile << "digraph prof {\n";
    for(auto dep = task_dependencies.begin(); dep != task_dependencies.end(); dep++) {
        task_identifier parent = dep->first;
        auto children = dep->second;
        string& parent_name = parent.get_name();
        for(auto offspring = children->begin(); offspring != children->end(); offspring++) {
            task_identifier child = offspring->first;
            int count = offspring->second;
            string& child_name = child.get_name();
            myfile << "  \"" << parent_name << "\" -> \"" << child_name << "\"";
            myfile << " [ label=\"  count: " << count << "\" ]; " << std::endl;
            
        }
    }
    myfile << "}\n";
    myfile.close();
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
  void profiler_listener::write_profile(int tid) {
    ofstream myfile;
    stringstream datname;
    map<string, profile*>* the_name_map = nullptr;
    map<apex_function_address, profile*>* the_address_map = nullptr;

    if (tid == -1) {
      the_name_map = &name_map;
      the_address_map = &address_map;
      // name format: profile.nodeid.contextid.threadid
      // We only write one profile per process
      datname << "profile." << node_id << ".0.0";
    } else {
      if (thread_name_maps[tid] == nullptr || thread_address_maps[tid] == nullptr) return;
      the_name_map = thread_name_maps[tid];
      the_address_map = thread_address_maps[tid];
      // name format: profile.nodeid.contextid.threadid
      datname << "profile." << node_id << ".0." << tid;
    }

    // name format: profile.nodeid.contextid.threadid
    myfile.open(datname.str().c_str());
    int counter_events = 0;

    // Determine number of counter events, as these need to be
    // excluded from the number of normal timers
    map<string, profile*>::const_iterator it2;
    for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == APEX_COUNTER) {
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
    map<apex_function_address, profile*>::const_iterator it;
    for(it = the_address_map->begin(); it != the_address_map->end(); it++) {
      profile * p = it->second;
      // ".TAU application" 1 8 8658984 8660739 0 GROUP="TAU_USER"
      apex_function_address function_address = it->first;
#if APEX_HAVE_BFD
      string * tmp = lookup_address((uintptr_t)function_address, true);
      myfile << "\"" << *tmp << "\" ";
#else
      myfile << "\"" << ti.map_addr_to_name(function_address) << "\" ";
#endif
      format_line (myfile, p);
    }

    // Iterate over the profiles which are associated to a function
    // by name. Only output the regular timers now. Counters are
    // in a separate section, below.
    profile * mainp = nullptr;
    double not_main = 0.0;
    for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
      profile * p = it2->second;
      if(p->get_type() == APEX_TIMER) {
        string action_name = it2->first;
        if(strcmp(action_name.c_str(), APEX_MAIN) == 0) {
          mainp = p;
        } else {
#if APEX_HAVE_BFD
      boost::regex rx (".*UNRESOLVED ADDR (.*)");
      if (boost::regex_match (action_name,rx)) {
        const boost::regex separator(" ADDR ");
        boost::sregex_token_iterator token(action_name.begin(), action_name.end(), separator, -1);
        *token++; // ignore
        string addr_str = *token++;
    void* addr_addr;
    sscanf(addr_str.c_str(), "%p", &addr_addr);
        string * tmp = lookup_address((uintptr_t)addr_addr, true);
        boost::regex old_address("UNRESOLVED ADDR " + addr_str);
    action_name = boost::regex_replace(action_name, old_address, *tmp);
      }
#endif
          myfile << "\"" << action_name << "\" ";
          format_line (myfile, p);
          not_main += p->get_accumulated();
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
      for(it2 = the_name_map->begin(); it2 != the_name_map->end(); it2++) {
        profile * p = it2->second;
        if(p->get_type() == APEX_COUNTER) {
          string action_name = it2->first;
          myfile << "\"" << action_name << "\" ";
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
   */
  void profiler_listener::process_profiles_wrapper(void) {
      apex::instance()->the_profiler_listener->process_profiles();
  }

  /* This is the main function for the consumer thread.
   * It will wait at a semaphore for pending work. When there is
   * work on one or more queues, it will iterate over the queues
   * and process the pending profiler objects, updating the profiles
   * as it goes. */
  void profiler_listener::process_profiles(void)
  {
    static bool _initialized = false;
    if (!_initialized) {
      initialize_worker_thread_for_TAU();
      _initialized = true;
    }
#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau()) {
      TAU_START("profiler_listener::process_profiles");
    }
#endif

    profiler* p;
    task_dependency* td;
    // Main loop. Stay in this loop unless "done".
#ifndef APEX_HAVE_HPX3
    while (!_done) {
      queue_signal.wait();
#endif
#ifdef APEX_HAVE_TAU
      /*
    if (apex_options::use_tau()) {
      TAU_START("profiler_listener::process_profiles: main loop");
    }
    */
#endif
      while(!_done && thequeue.try_dequeue(p)) {
        process_profile(p, 0);
      }
      if (apex_options::use_taskgraph_output()) {
        while(!_done && dependency_queue.try_dequeue(td)) {
          process_dependency(td);
        }
      }
      /* 
       * I want to process the tasks concurrently, but this loop
       * is too much overhead. Maybe dequeue them in batches?
       */
      /*
      std::vector<std::future<unsigned int>> pending_futures;
      while(!_done && thequeue.try_dequeue(p)) {
          auto f = std::async(process_profile,p,0);
          // transfer the future's shared state to a longer-lived future
          pending_futures.push_back(std::move(f));
      }
      std::vector<std::future<unsigned int>>::iterator iter;
      for (iter = pending_futures.begin() ; iter < pending_futures.end() ; iter++ ) {
          iter->get();
      }
      */

#ifdef USE_UDP
      // are we updating a global profile?
      if (apex_options::use_udp_sink()) {
          udp_client::synchronize_profiles(name_map, address_map);
      }
#endif
#ifdef APEX_HAVE_TAU
      /*
      if (apex_options::use_tau()) {
        TAU_STOP("profiler_listener::process_profiles: main loop");
      }
      */
#endif
#ifndef APEX_HAVE_HPX3
    }

    size_t ignored = thequeue.size_approx();
    if (ignored > 0) {
      std::cerr << "Info: " << ignored << " items remaining on on the profiler_listener queue...";
    }
    // We might be done, but check to make sure the queue is empty
    while(!_done && thequeue.try_dequeue(p)) {
    //while(thequeue.try_dequeue(p)) {
      process_profile(p, 0);
    }
    if (ignored > 0) {
      std::cerr << "done." << std::endl;
    }
    if (apex_options::use_taskgraph_output()) {
      // process the task dependencies
      while(dependency_queue.try_dequeue(td)) {
        process_dependency(td);
      }
    }
 
    // stop the main timer, and process that profile
    //main_timer->stop();
    //process_profile(main_timer, my_tid);

#ifdef USE_UDP
    // are we updating a global profile?
    if (apex_options::use_udp_sink()) {
        udp_client::synchronize_profiles(name_map, address_map);
        udp_client::stop_client();
    }
#endif

#endif // NOT DEFINED APEX_HAVE_HPX3

#ifdef APEX_HAVE_HPX3
    consumer_task_running.clear(memory_order_release);
#endif

#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau()) {
      TAU_STOP("profiler_listener::process_profiles");
    }
#endif
  }

#ifdef APEX_HAVE_HPX3
} // end namespace apex (HPX_PLAIN_ACTION needs to be in global namespace)

HPX_PLAIN_ACTION(profiler_listener::process_profiles, apex_internal_process_profiles_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(apex_internal_process_profiles_action);

namespace apex {

void profiler_listener::schedule_process_profiles() {
    if(get_hpx_runtime_ptr() == nullptr) return;
    if(hpx_shutdown) {
        profiler_listener::process_profiles();
    } else if(!consumer_task_running.test_and_set(memory_order_acq_rel)) {
        apex_internal_process_profiles_action act;
        try {
            hpx::apply(act, hpx::find_here());
        } catch(...) {
            // During shutdown, we can't schedule a new task,
            // so we process profiles ourselves.
            profiler_listener::process_profiles();
        }
    } 
}
#endif

#if APEX_HAVE_PAPI
APEX_NATIVE_TLS int EventSet = PAPI_NULL;
#define PAPI_ERROR_CHECK(name) \
if (rc != 0) cout << "name: " << rc << ": " << PAPI_strerror(rc) << endl;

  void profiler_listener::initialize_PAPI(bool first_time) {
      int rc = 0;
      if (first_time) {
        PAPI_library_init( PAPI_VER_CURRENT );
        rc = PAPI_multiplex_init(); // use more counters than allowed
        PAPI_ERROR_CHECK(PAPI_multiplex_init);
        PAPI_thread_init( thread_instance::get_id );
        rc = PAPI_set_domain(PAPI_DOM_ALL);
        PAPI_ERROR_CHECK(PAPI_set_domain);
      }
      rc = PAPI_create_eventset(&EventSet);
      PAPI_ERROR_CHECK(PAPI_create_eventset);
      rc = PAPI_assign_eventset_component (EventSet, 0);
      PAPI_ERROR_CHECK(PAPI_assign_eventset_component);
      rc = PAPI_set_granularity(PAPI_GRN_THR);
      PAPI_ERROR_CHECK(PAPI_set_granularity);
      rc = PAPI_set_multiplex(EventSet);
      PAPI_ERROR_CHECK(PAPI_set_multiplex);
      // parse the requested set of papi counters
      // The string is modified by strtok, so copy it.
      if (strlen(apex_options::papi_metrics()) > 0) {
        char* tmpstr = strdup(apex_options::papi_metrics());
        char *p = strtok(tmpstr, " ");
        int code;
        while (p) {
          printf ("Trying PAPI Metric: %s\n", p);
          int rc = PAPI_event_name_to_code(p, &code);
          if (PAPI_query_event (code) == PAPI_OK) {
            rc = PAPI_add_event(EventSet, code);
            PAPI_ERROR_CHECK(PAPI_add_event);
            metric_names.push_back(string(p));
            num_papi_counters++;
          }
          p = strtok(NULL, " ");
        }
        rc = PAPI_start( EventSet );
        PAPI_ERROR_CHECK(PAPI_start);
      }
  }

#endif

  /* When APEX gets a STARTUP event, do some initialization. */
  void profiler_listener::on_startup(startup_event_data &data) {
    if (!_done) {
      if (apex_options::use_profile_output() > 1) {
        thread_address_maps[0] = new map<apex_function_address, profile*>();
        thread_name_maps[0] = new map<string, profile*>();
      }
#ifndef APEX_HAVE_HPX3
      // Start the consumer thread, to process profiler objects.
      consumer_thread = new boost::thread(process_profiles_wrapper);
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
      //main_timer = std::shared_ptr<profiler>(new profiler(new string(APEX_MAIN)));
    }
    APEX_UNUSED(data);
  }

  /* On the shutdown event, notify the consumer thread that we are done
   * and set the "terminate" flag. */
  void profiler_listener::on_shutdown(shutdown_event_data &data) {
    if (_done) { return; }
    if (!_done) {
      _done = true;
      node_id = data.node_id;
      //sleep(1);
#ifndef APEX_HAVE_HPX3
      queue_signal.post();
      if (consumer_thread != nullptr) {
          consumer_thread->join();
      }
#endif

      // output to screen?
      if (apex_options::use_screen_output() && node_id == 0)
      {
        finalize_profiles();
      }
      if (apex_options::use_taskgraph_output() && node_id == 0)
      {
        write_taskgraph();
      }

      // output to 1 TAU profile per process?
      if (apex_options::use_profile_output() == 1)
      {
        write_profile(-1);
      }
      // output to TAU profiles, one per thread per process?
      else if (apex_options::use_profile_output() > 1)
      {
        // the number of thread_name_maps tells us how many threads there are to process
        for (unsigned int i = 0 ; i < thread_name_maps.size(); i++) {
          write_profile((int)i);
        }
      }

#if APEX_HAVE_PAPI
      if (num_papi_counters > 0) {
        int rc = 0;
        int i = 0;
        long long values[8] {0L};
        //cout << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << endl;
        for (i = 0 ; i < thread_instance::get_num_threads() ; i++) {
          rc = PAPI_accum( event_sets[i], values );
          PAPI_ERROR_CHECK(PAPI_stop);
          //cout << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << endl;
        }
        if (apex_options::use_screen_output() && node_id == 0 && num_papi_counters > 0) {
          cout << endl << "TOTAL COUNTERS for " << thread_instance::get_num_threads() << " threads:" << endl;
          for (i = 0 ; i < num_papi_counters ; i++) {
            cout << metric_names[i] << " : " << values[i] << endl;
          }
          cout << endl;
        }
      }
#endif
    }
    /* The cleanup is disabled for now. Why? Because we want to be able
     * to access the profiles at the end of the run, after APEX has
     * finalized. */
    // cleanup.
    // delete_profiles();
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
      // resize the vector
      if (apex_options::use_profile_output() > 1) {
        _mtx.lock();
        if (my_tid >= thread_address_maps.size()) {
            thread_address_maps.resize(my_tid + 1);
            thread_address_maps[my_tid] = new map<apex_function_address, profile*>();
        }
        if (my_tid >= thread_name_maps.size()) {
            thread_name_maps.resize(my_tid + 1);
            thread_name_maps[my_tid] = new map<string, profile*>();
        }
        _mtx.unlock();
      }
#if APEX_HAVE_PAPI
      initialize_PAPI(false);
      if (my_tid >= event_sets.size()) {
        event_sets.resize(my_tid + 1);
      }
      event_sets[my_tid] = EventSet;
#endif
    }
    APEX_UNUSED(data);
  }

  extern "C" int main (int, char**);

  /* When a start event happens, create a profiler object. Unless this
   * named event is throttled, in which case do nothing, as quickly as possible */
  inline bool profiler_listener::_common_start(apex_function_address function_address, bool is_resume) {
    if (!_done) {
#if defined(APEX_THROTTLE)
        // if this timer is throttled, return without doing anything
        unordered_set<apex_function_address>::const_iterator it = throttled_addresses.find(function_address);
        if (it != throttled_addresses.end()) {
          /*
           * The throw is removed, because it is a performance penalty on some systems
           * on_start now returns a boolean
           */
          //throw disabled_profiler_exception(); // to be caught by apex::start/resume
          return false;
        }
#endif
#if APEX_HAVE_TAU
        // create a dummy profiler object, without start timestamp.
        thread_instance::instance().set_current_profiler(std::shared_ptr<profiler>(new profiler(function_address, 0.0)));
#else
        // start the profiler object, which starts our timers
        thread_instance::instance().set_current_profiler(std::shared_ptr<profiler>(new profiler(function_address, is_resume)));
#endif
#if APEX_HAVE_PAPI
        long long * values = thread_instance::instance().get_current_profiler()->papi_start_values;
        int rc = 0;
        rc = PAPI_read( EventSet, values );
        PAPI_ERROR_CHECK(PAPI_read);
#endif
    } else {
        return false;
    }
    return true;
  }

  /* When a start event happens, create a profiler object. Unless this
   * named event is throttled, in which case do nothing, as quickly as possible */
  inline bool profiler_listener::_common_start(std::string * timer_name, bool is_resume) {
    if (!_done) {
#if defined(APEX_THROTTLE)
      // if this timer is throttled, return without doing anything
      unordered_set<string>::const_iterator it = throttled_names.find(*timer_name);
      if (it != throttled_names.end()) {
          /*
           * The throw is removed, because it is a performance penalty on some systems
           * on_start now returns a boolean
           */
        //throw disabled_profiler_exception(); // to be caught by apex::start/resume
        return false;
      }
#endif
#if APEX_HAVE_TAU
      // create a dummy profiler object, without start timestamp.
      thread_instance::instance().set_current_profiler(std::shared_ptr<profiler>(new profiler(timer_name, 0.0)));
#else
      // start the profiler object, which starts our timers
      thread_instance::instance().set_current_profiler(std::shared_ptr<profiler>(new profiler(timer_name, is_resume)));
#endif
#if APEX_HAVE_PAPI
      long long * values = thread_instance::instance().get_current_profiler()->papi_start_values;
      int rc = 0;
      rc = PAPI_read( EventSet, values );
      PAPI_ERROR_CHECK(PAPI_read);
#endif
    } else {
        return false;
    }
    return true;
  }

  inline void profiler_listener::push_profiler(int my_tid, std::shared_ptr<profiler>p) {
      // we have to make a local copy, because lockfree queues DO NOT SUPPORT shared_ptrs!
      profiler* local_p = new profiler(p.get());
      bool worked = thequeue.enqueue(local_p);
      if (!worked) {
          static bool issued = false;
          if (!issued) {
              issued = true;
              if(p->have_name) {
                  cout << "APEX Warning : failed to push " << *(p->timer_name) << endl;
              } else {
#if defined(HAVE_BFD)
                  cout << "APEX Warning : failed to push " << *(lookup_address((uintptr_t)p->action_address, true) << endl;
#else
                  cout << "APEX Warning : failed to push address " << p->action_address << endl;
#endif
              }
              cout << "One or more frequently-called, lightweight functions is being timed." << endl;
          }
      }
#ifndef APEX_HAVE_HPX3
      queue_signal.post();
#endif
#ifdef APEX_HAVE_HPX3
      schedule_process_profiles();
#endif
  }

  /* Stop the timer, if applicable, and queue the profiler object */
  inline void profiler_listener::_common_stop(std::shared_ptr<profiler> p, bool is_yield) {
    if (!_done) {
      if (p) {
        p->stop(is_yield);
#if APEX_HAVE_PAPI
        long long * values = p->papi_stop_values;
        int rc = 0;
        rc = PAPI_read( EventSet, values );
        PAPI_ERROR_CHECK(PAPI_read);
#endif
        if (apex_options::use_taskgraph_output()) {
          if (!p->is_resume) { 
            // get the PARENT profiler
            std::shared_ptr<profiler> parent_profiler = nullptr;
            try {
              parent_profiler = thread_instance::instance().get_parent_profiler();
              task_identifier * parent = new task_identifier(parent_profiler.get());
              task_identifier * child = new task_identifier(p.get());
              dependency_queue.enqueue(new task_dependency(parent, child));
            } catch (empty_stack_exception& e) { }
          }
        }
        push_profiler(my_tid, p);
      }
    }
  }

  /* Start the timer */
  bool profiler_listener::on_start(apex_function_address function_address) {
    return _common_start(function_address, false);
  }

  /* Start the timer */
  bool profiler_listener::on_start(std::string * timer_name) {
    return _common_start(timer_name, false);
  }

  /* This is just like starting a timer, but don't increment the number of calls
   * value. That is because we are restarting an existing timer. */
  bool profiler_listener::on_resume(std::string * timer_name) {
        return _common_start(timer_name, true);
  }

  /* This is just like starting a timer, but don't increment the number of calls
   * value. That is because we are restarting an existing timer. */
  bool profiler_listener::on_resume(apex_function_address function_address) {
        return _common_start(function_address, true);
  }

   /* Stop the timer */
  void profiler_listener::on_stop(std::shared_ptr<profiler> p) {
// if we aren't using TAU, process the profiler object.
#ifndef APEX_HAVE_TAU
    _common_stop(p, p->is_resume); // don't change the yield/resume value!
#endif
  }

  /* Stop the timer, but don't increment the number of calls */
  void profiler_listener::on_yield(std::shared_ptr<profiler> p) {
// if we aren't using TAU, process the profiler object.
#ifndef APEX_HAVE_TAU
    _common_stop(p, true);
#endif
  }

  /* When a thread exits, pop and stop all timers. */
  void profiler_listener::on_exit_thread(event_data &data) {
    APEX_UNUSED(data);
  }

  /* When a sample value is processed, save it as a profiler object, and queue it. */
  void profiler_listener::on_sample_value(sample_value_event_data &data) {
#ifndef APEX_HAVE_TAU
    if (!_done) {
        std::shared_ptr<profiler> p = std::shared_ptr<profiler>(new profiler(new string(*data.counter_name), data.counter_value));
      p->is_counter = data.is_counter;
      push_profiler(my_tid, p);
    }
#endif
  }

  void profiler_listener::on_new_task(apex_function_address function_address, void * task_id) {
    if (!apex_options::use_taskgraph_output()) { return; }
    // get the current profiler
    std::shared_ptr<profiler> p = thread_instance::instance().get_current_profiler();
    if (p != NULL) {
        task_identifier * parent = new task_identifier(p.get());
        task_identifier * child = new task_identifier(function_address);
        dependency_queue.enqueue(new task_dependency(parent, child));
    } else {
        task_identifier * parent = new task_identifier(string("__start"));
        task_identifier * child = new task_identifier(function_address);
        dependency_queue.enqueue(new task_dependency(parent, child));
    }
  }

  void profiler_listener::on_new_task(std::string *timer_name, void * task_id) {
    if (!apex_options::use_taskgraph_output()) { return; }
    // get the current profiler
    std::shared_ptr<profiler> p = thread_instance::instance().get_current_profiler();
    if (p != NULL) {
        task_identifier * parent = new task_identifier(p.get());
        task_identifier * child = new task_identifier(*timer_name);
        dependency_queue.enqueue(new task_dependency(parent, child));
    } else {
        task_identifier * parent = new task_identifier(string("__start"));
        task_identifier * child = new task_identifier(*timer_name);
        dependency_queue.enqueue(new task_dependency(parent, child));
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

  void profiler_listener::reset(apex_function_address function_address) {
      std::shared_ptr<profiler> p;
    if(function_address != APEX_NULL_FUNCTION_ADDRESS) {
    p = std::shared_ptr<profiler>(new profiler(function_address, false, reset_type::CURRENT));
    } else {
    p = std::shared_ptr<profiler>(new profiler((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, false, reset_type::ALL));
    }
    push_profiler(my_tid, p);
  }

  void profiler_listener::reset(const std::string &timer_name) {
      std::shared_ptr<profiler> p;
    p = std::shared_ptr<profiler>(new profiler(new string(timer_name), false, reset_type::CURRENT));
    push_profiler(my_tid, p);
  }

  profiler_listener::~profiler_listener (void) { 
      _done = true; // yikes!
      finalize();
#ifdef USE_UDP
      if (apex_options::use_udp_sink()) {
          udp_client::stop_client();
      }
#endif
      delete_profiles();
#ifndef APEX_HAVE_HPX3
      delete consumer_thread;
#endif
  };

}
