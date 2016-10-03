//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include "profiler.hpp"
#include <exception>
#ifdef APEX_DEBUG
#include <unordered_set>
#endif

namespace apex {

    /*
class empty_stack_exception : public std::exception {
    virtual const char* what() const throw() {
        return "Empty profiler stack.";
    }
} ;
*/

class thread_instance {
private:
  // TAU id of the thread
  int _id;
  // Runtime id of the thread
  int _runtime_id;
  // "name" of the thread
  std::string _top_level_timer_name;
  // is this an HPX worker thread?
  bool _is_worker;
  // map from name to thread id - common to all threads
  static std::map<std::string, int> _name_map;
  static std::mutex _name_map_mutex;
  // map from thread id to is_worker
  static std::map<int, bool> _worker_map;
  static std::mutex _worker_map_mutex;
  static std::atomic_int _num_threads;
  static std::atomic_int _active_threads;
  static std::string * _program_path;
  // thread specific data
  static APEX_NATIVE_TLS thread_instance * _instance;
  // constructor
  thread_instance (void) : _id(-1), _runtime_id(-1), _top_level_timer_name(), _is_worker(false) { };
  // map from function address to name - unique to all threads to avoid locking
  std::map<apex_function_address, std::string> _function_map;
  profiler * current_profiler;
  /*
  std::vector<profiler*> current_profilers;
  */
  //std::shared_ptr<profiler> current_profiler;
  //std::vector<std::shared_ptr<profiler> > current_profilers;
public:
  ~thread_instance(void);
  static thread_instance& instance(void);
  static long unsigned int get_id(void) { return instance()._id; }
  static long unsigned int get_runtime_id(void) { return instance()._runtime_id; }
  static void set_runtime_id(long unsigned int id) { instance()._runtime_id = id; }
  static std::string get_name(void);
  static void set_name(std::string name);
  static void set_worker(bool is_worker);
  static int map_name_to_id(std::string name);
  static bool map_id_to_worker(int id);
  static int get_num_threads(void) { return _num_threads; };
  std::string map_addr_to_name(apex_function_address function_address);
  //static void set_current_profiler(std::shared_ptr<profiler> &the_profiler);
  //static std::shared_ptr<profiler> get_current_profiler(void);
  //static std::shared_ptr<profiler> get_parent_profiler(void);
  //static std::shared_ptr<profiler> pop_current_profiler(void);
  //static std::shared_ptr<profiler> pop_current_profiler(profiler * requested);
  static void set_current_profiler(profiler * the_profiler);
  static profiler * get_current_profiler(void);
  static void clear_current_profiler(void);
  /*
  static profiler * get_parent_profiler(void);
  static profiler * pop_current_profiler(void);
  static profiler * pop_current_profiler(profiler * requested);
  static bool profiler_stack_empty(void);
  */
  static const char * program_path(void);
#ifdef APEX_DEBUG
  static std::mutex _open_profiler_mutex;
  static std::unordered_set<std::string> open_profilers;
  static void add_open_profiler(profiler* p) {
      std::unique_lock<std::mutex> l(_open_profiler_mutex);
      std::stringstream ss;
      ss << p->task_id->get_name();
      ss << p->time_point_to_nanoseconds(p->start);
      open_profilers.insert(ss.str());
  }
  static void remove_open_profiler(int id, profiler *p) {
      if (p == NULL) return;
      std::unique_lock<std::mutex> l(_open_profiler_mutex);
      std::stringstream ss;
      ss << p->task_id->get_name();
      ss << p->time_point_to_nanoseconds(p->start);
      auto tmp = open_profilers.find(ss.str());
      if (tmp != open_profilers.end()) {
        open_profilers.erase(ss.str());
      } else {
        std::cout << id << ": Warning! Can't find open profiler: " << ss.str() << std::endl;fflush(stdout);
      }
  }
  static std::unordered_set<std::string>& get_open_profilers(void) {
      return open_profilers;
  }
#endif
};

}

