//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_api.hpp" // make this the first include.
#include "thread_instance.hpp"
#include <iostream>

// TAU related
#ifdef APEX_HAVE_TAU
#define PROFILING_ON
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif

#include <stdio.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
#include <execinfo.h>
#include <linux/limits.h>
#elif __APPLE__
#  include <mach-o/dyld.h>
#elif defined(__FreeBSD__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif

#ifdef APEX_HAVE_BFD
#include "address_resolution.hpp"
#endif

using namespace std;

namespace apex {

// Global static pointer used to ensure a single instance of the class.
APEX_NATIVE_TLS thread_instance * thread_instance::_instance(nullptr);
// Global static count of threads in system
std::atomic_int thread_instance::_num_threads(0);
// Global static count of *active* threads in system
std::atomic_int thread_instance::_active_threads(0);
// Global static map of HPX thread names to TAU thread IDs
map<string, int> thread_instance::_name_map;
// Global static mutex to control access to the map
std::mutex thread_instance::_name_map_mutex;
// Global static map of TAU thread IDs to HPX workers
map<int, bool> thread_instance::_worker_map;
// Global static mutex to control access to the map
std::mutex thread_instance::_worker_map_mutex;
// Global static path to executable name
string * thread_instance::_program_path = nullptr;

thread_instance& thread_instance::instance(void) {
  thread_instance* me = _instance;
  if( ! me ) {
    // first time called by this thread
    // construct test element to be used in all subsequent calls from this thread
    _instance = new thread_instance();
    me = _instance;
    //me->_id = TAU_PROFILE_GET_THREAD();
    me->_id = _num_threads++;
    _active_threads++;
  }
  return *me;
}

thread_instance::~thread_instance(void) {
    if (_id == 0) {
        finalize();
    }
    _active_threads--;
}

void thread_instance::set_worker(bool is_worker) {
  instance()._is_worker = is_worker;
  std::unique_lock<std::mutex> l(_worker_map_mutex);
  _worker_map[get_id()] = is_worker;
}

string thread_instance::get_name(void) {
  return instance()._top_level_timer_name;
}

void thread_instance::set_name(string name) {
  if (instance()._top_level_timer_name.empty())
  {
    instance()._top_level_timer_name = name;
    {
        std::unique_lock<std::mutex> l(_name_map_mutex);
        _name_map[name] = get_id();
    }
    if (name.find("worker-thread") != string::npos) {
      instance().set_worker(true);
    }
  }
}

int thread_instance::map_name_to_id(string name) {
  //cout << "Looking for " << name << endl;
  int tmp = -1;
  {
    std::unique_lock<std::mutex> l(_name_map_mutex);
    map<string, int>::const_iterator it = _name_map.find(name);
    if (it != _name_map.end()) {
      tmp = (*it).second;
    }
  }
  return tmp;
}

bool thread_instance::map_id_to_worker(int id) {
  //cout << "Looking for " << name << endl;
  bool worker = false;
  {
    std::unique_lock<std::mutex> l(_worker_map_mutex);
    map<int, bool>::const_iterator it = _worker_map.find(id);
    if (it != _worker_map.end()) {
      worker = (*it).second;
    }
  }
  return worker;
}

const char* thread_instance::program_path(void) {

#if defined(_WIN32) || defined(_WIN64)

    if (_program_path == NULL) {
        char path[MAX_PATH + 1] = { '\0' };
        if (!GetModuleFileName(NULL, path, sizeof(path)))
            return NULL;
        _program_path = new string(path);
    }
    return _program_path->c_str();

#elif defined(__linux) || defined(linux) || defined(__linux__)

    if (_program_path == NULL) {
        char path[PATH_MAX];
        memset(path,0,PATH_MAX);
        if (path != NULL) {
            if (readlink("/proc/self/exe", path, PATH_MAX) == -1)
                return NULL;
        }
        _program_path = new string(path);
    }
    return _program_path->c_str();

#elif defined(__APPLE__)

    if (_program_path == NULL) {
        char path[PATH_MAX + 1];
        std::uint32_t len = sizeof(path) / sizeof(path[0]);

        if (0 != _NSGetExecutablePath(path, &len))
            return NULL;

        path[len] = '\0';
        _program_path = new string(path);
    }
    return _program_path->c_str();

#elif defined(__FreeBSD__)

    if (_program_path == NULL) {
        int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
        size_t cb = 0;
        sysctl(mib, 4, NULL, &cb, NULL, 0);
        if (!cb)
            return NULL;

        std::vector<char> buf(cb);
        sysctl(mib, 4, &buf[0], &cb, NULL, 0);
        _program_path = new string(&buf[0]);
    }
    return _program_path->c_str();

#else
#  error Unsupported platform
#endif
    return NULL;
}

string thread_instance::map_addr_to_name(apex_function_address function_address) {
  auto it = _function_map.find(function_address);
  if (it != _function_map.end()) {
    return (*it).second;
  } // else...
#ifdef APEX_HAVE_BFD
  string * name = lookup_address(function_address, false);
  _function_map[function_address] = *name;
  return *name;
#else
  stringstream ss;
  const char * progname = program_path();
  if (progname == NULL) {
    ss << "UNRESOLVED  ADDR 0x" << hex << function_address;
  } else {
    ss << "UNRESOLVED " << string(progname) << " ADDR " << hex << function_address;
  }
  string name = string(ss.str());
  _function_map[function_address] = name;
  return name;
#endif
}

void thread_instance::set_current_profiler(profiler * the_profiler) {
    instance().current_profiler = the_profiler;
    /*
    instance().current_profilers.push_back(the_profiler);
    */
}

profiler * thread_instance::get_current_profiler(void) {
    return instance().current_profiler;
}

profiler * thread_instance::get_parent_profiler(void) {
    if (instance().current_profilers.size() == 0) {
        //throw empty_stack_exception(); // to be caught by the profiler_listener
        return nullptr;
    }
    return instance().current_profilers.back();
}

profiler * thread_instance::pop_current_profiler(void) {
    /*
    if (instance().current_profilers.empty()) {
        //throw empty_stack_exception(); // to be caught by the profiler_listener
        return nullptr;
    }
    instance().current_profiler = instance().current_profilers.back();
    instance().current_profilers.pop_back();
    */
    return instance().current_profiler;
}

bool thread_instance::profiler_stack_empty() {
    return instance().current_profilers.empty();
}

profiler * thread_instance::pop_current_profiler(profiler * requested) {
    if (instance().current_profilers.empty()) {
        //throw empty_stack_exception(); // to be caught by the profiler_listener
        return requested;
    }
    if (instance().current_profilers.back() == requested) {
      instance().current_profiler = instance().current_profilers.back();
      instance().current_profilers.pop_back();
    } else {
      // work backward over the vector to find the requested profiler
      std::vector<profiler*>::const_iterator it;
      int crappy_compiler = 0;
      for (it = instance().current_profilers.end() ; it != instance().current_profilers.begin() ; it-- ) {
        profiler * tmp = (*it);
        if (tmp == requested) {
          instance().current_profiler = *it;
//#ifdef __INTEL_COMPILER
#if __cplusplus < 201400L
          instance().current_profilers.erase(instance().current_profilers.end()-crappy_compiler);
#else
          instance().current_profilers.erase(it);
#endif
          return instance().current_profiler;
        }
        crappy_compiler++;
      }
      //throw empty_stack_exception(); // to be caught by the profiler_listener
      return requested;
    }
    return instance().current_profiler; // for completeless
}

}
