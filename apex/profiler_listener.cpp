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

#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>
#include <unistd.h>
#include <sched.h>
#include <cstdio>

//#define MAX_QUEUE_SIZE 1024*1024
#define MAX_QUEUE_SIZE 128

using namespace std;

namespace apex {

boost::atomic<int> profiler_listener::active_tasks(0);

std::vector<boost::lockfree::spsc_queue<profiler*>* > profiler_queues(8);
boost::thread * consumer_thread;
boost::atomic<bool> done (false);
semaphore queue_signal;
static map<string, profile*> name_map;
static map<void*, profile*> address_map;

profiler * profiler_listener::main_timer(NULL);
int profiler_listener::node_id(0);

profile* profiler_listener::get_profile(void * address) {
   map<void*, profile*>::iterator it = address_map.find(address);
   if (it != address_map.end()) {
     return (*it).second;
   }
   return NULL;
}

profile* profiler_listener::get_profile(string &timer_name) {
   map<string, profile*>::iterator it = name_map.find(timer_name);
   if (it != name_map.end()) {
     return (*it).second;
   }
   return NULL;
}

inline void profiler_listener::process_profile(profiler * p)
{
    profile * theprofile;
    if (p->have_name) {
      map<string, profile*>::iterator it = name_map.find(*(p->timer_name));
      if (it != name_map.end()) {
        theprofile = (*it).second;
        theprofile->increment(p->elapsed());
      } else {
        theprofile = new profile(p->elapsed());
        name_map[*(p->timer_name)] = theprofile;
	// done with the name now
	delete(p->timer_name);
      }
    } else {
      map<void*, profile*>::const_iterator it2 = address_map.find(p->action_address);
      if (it2 != address_map.end()) {
        theprofile = (*it2).second;
        theprofile->increment(p->elapsed());
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
    int i = 0;
    for (i = 0 ; i < thread_instance::get_num_threads(); i++) {
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
      cout << function_address << ": " ;
      cout << p->get_calls() << ", " ;
      cout << p->get_minimum() << ", " ;
      cout << p->get_mean() << ", " ;
      cout << p->get_maximum() << ", " ;
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

void profiler_listener::write_profile(void) {
    ofstream myfile;
    stringstream datname;
    datname << "profile." << node_id << ".0.0";
    myfile.open(datname.str().c_str());
    int function_count = address_map.size() + name_map.size();
    // 1504 templated_functions_MULTI_TIME
    myfile << function_count << " templated_functions_MULTI_TIME" << endl;
    // # Name Calls Subrs Excl Incl ProfileCalls #
    myfile << "# Name Calls Subrs Excl Incl ProfileCalls #" << endl;
    thread_instance ti = thread_instance::instance();
    map<void*, profile*>::const_iterator it;
    for(it = address_map.begin(); it != address_map.end(); it++) {
      profile * p = it->second;
      // ".TAU application" 1 8 8658984 8660739 0 GROUP="TAU_USER"
      void * function_address = it->first;
      myfile << "\"" << ti.map_addr_to_name(function_address) << "\" ";
      format_line (myfile, p);
    }
    map<string, profile*>::const_iterator it2;
    for(it2 = name_map.begin(); it2 != name_map.end(); it2++) {
      profile * p = it2->second;
      string action_name = it2->first;
      myfile << "\"" << action_name << "\" ";
      format_line (myfile, p);
    }
    // 0 aggregates
    myfile << "0 aggregates" << endl;
    myfile.close();
}

void profiler_listener::process_profiles(void)
{
    profiler * p;
    int i;
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

    char* option = NULL;
    option = getenv("APEX_SCREEN_OUTPUT");
    if (option != NULL && strcmp(option,"1") == 0)
    {
      finalize_profiles();
    }
    option = getenv("APEX_PROFILE_OUTPUT");
    if (option != NULL && strcmp(option,"1") == 0)
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
      unsigned int me = (unsigned int)thread_instance::instance().get_id();
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

void profiler_listener::on_start(timer_event_data &data) {
    static __thread int counter = 0; // only do 1/10 of the timers
    //if (counter++ % 10 != 0) { return; }
  if (!_terminate) {
      //active_tasks++;
      //cout << "START " << active_tasks << " " <<  *(data.timer_name) << " " << data.function_address << endl;
      if (data.have_name) {
        data.my_profiler = new profiler(data.timer_name);
      } else {
        data.my_profiler = new profiler(data.function_address);
      }
  }
}

void profiler_listener::on_stop(timer_event_data &data) {
  static __thread int counter = 0; // only do 1/10 of the timers
  if (!_terminate) {
      //active_tasks--;
      //cout << "STOP " << active_tasks << " " << endl;
      if (data.my_profiler) {
          data.my_profiler->stop();
          int me = thread_instance::instance().get_id();
          profiler_queues[me]->push(data.my_profiler);
          if (counter++ % 50 == 0) { // after there are some timers, post.
      	    queue_signal.post();
          }
      }
  }
}

void profiler_listener::on_resume(timer_event_data &data) {
  if (!_terminate) {
      active_tasks++;
      //cout << "RESUME " << active_tasks << " " <<  *(data.timer_name) << " " << data.function_address << endl;
      if (data.have_name) {
        data.my_profiler = new profiler(data.timer_name);
      } else {
        data.my_profiler = new profiler(data.function_address);
      }
  }
}

void profiler_listener::on_sample_value(sample_value_event_data &data) {
  if (!_terminate) {
      //cout << "SAMPLE VALUE" << endl;
  }
}

void profiler_listener::on_periodic(periodic_event_data &data) {
  if (!_terminate) {
      //cout << "PERIODIC" << endl;
  }
}

}
