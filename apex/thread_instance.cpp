//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "thread_instance.hpp"
#include <iostream>

// TAU related
#ifdef APEX_HAVE_TAU
#define PROFILING_ON
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif

using namespace std;

namespace apex {

// Global static pointer used to ensure a single instance of the class.
boost::thread_specific_ptr<thread_instance> thread_instance::_instance;
// Global static count of threads in system
boost::atomic_int thread_instance::_num_threads(0);
// Global static map of HPX thread names to TAU thread IDs
map<string, int> thread_instance::_name_map;
// Global static mutex to control access to the map
boost::mutex thread_instance::_name_map_mutex;
// Global static map of TAU thread IDs to HPX workers
map<int, bool> thread_instance::_worker_map;
// Global static mutex to control access to the map
boost::mutex thread_instance::_worker_map_mutex;

thread_instance* thread_instance::instance(void) {
  thread_instance* me = _instance.get();
  if( ! me ) {
    // first time called by this thread
    // construct test element to be used in all subsequent calls from this thread
    _instance.reset( new thread_instance());
    me = _instance.get();
    //me->_id = TAU_PROFILE_GET_THREAD();
    me->_id = _num_threads++;
  }
  return me;
}

int thread_instance::get_id(void) {
  return instance()->_id;
}

void thread_instance::set_worker(bool is_worker) {
  instance()->_is_worker = is_worker;
  _worker_map_mutex.lock();
  _worker_map[instance()->get_id()] = is_worker;
  _worker_map_mutex.unlock();
}

string thread_instance::get_name(void) {
  return *(instance()->_top_level_timer_name);
}

void thread_instance::set_name(string name) {
  if (instance()->_top_level_timer_name == NULL)
  {
    instance()->_top_level_timer_name = new string(name);
    _name_map_mutex.lock();
    _name_map[name] = instance()->get_id();
    _name_map_mutex.unlock();
    if (name.find("worker-thread") != name.npos) {
      instance()->set_worker(true);
    }
  }
}

int thread_instance::map_name_to_id(string name) {
  //cout << "Looking for " << name << endl;
  int tmp = -1;
  _name_map_mutex.lock();
  if (_name_map.find(name) != _name_map.end()) {
    tmp = _name_map[name];
  }
  _name_map_mutex.unlock();
  return tmp;
}

bool thread_instance::map_id_to_worker(int id) {
  //cout << "Looking for " << name << endl;
  bool worker = false;
  _worker_map_mutex.lock();
  if (_worker_map.find(id) != _worker_map.end()) {
    worker = _worker_map[id];
  }
  _worker_map_mutex.unlock();
  return worker;
}

}
