//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef THREADINSTANCE_H
#define THREADINSTANCE_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <strings.h>
#include <boost/thread/tss.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/atomic.hpp>
#include <map>

//using namespace std;

namespace apex {

class thread_instance {
private:
  // TAU id of the thread
  int _id;
  // "name" of the thread
  std::string *_top_level_timer_name;
  // is this an HPX worker thread?
  bool _is_worker;
  // map from name to thread id - common to all threads
  static std::map<std::string, int> _name_map;
  static boost::mutex _name_map_mutex;
  // map from thread id to is_worker
  static std::map<int, bool> _worker_map;
  static boost::mutex _worker_map_mutex;
  static boost::atomic_int _num_threads;
  // thread specific data
  static boost::thread_specific_ptr<thread_instance> _instance;
  // constructor
  thread_instance (void) : _id(-1), _top_level_timer_name(NULL), _is_worker(false) { };
public:
  static thread_instance* instance(void);
  static int get_id(void);
  static std::string get_name(void);
  static void set_name(std::string name);
  static void set_worker(bool is_worker);
  static int map_name_to_id(std::string name);
  static bool map_id_to_worker(int id);
  static int get_num_threads(void) { return _num_threads; };
};

}

#endif // THREADINSTANCE_H
