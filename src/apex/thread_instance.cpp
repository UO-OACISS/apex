//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_api.hpp" // make this the first include.
#include "apex.hpp"
#include "thread_instance.hpp"
#include <iostream>

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

#include <assert.h>

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
// Global static mutex to control access to the hashmap of resolved addresses
shared_mutex_type thread_instance::_function_map_mutex;
// Global static map of TAU thread IDs to HPX workers
map<int, bool> thread_instance::_worker_map;
// Global static mutex to control access to the map
std::mutex thread_instance::_worker_map_mutex;
// Global static path to executable name
string * thread_instance::_program_path = nullptr;
#ifdef APEX_DEBUG
// Global static mutex to control access for debugging purposes
std::mutex thread_instance::_open_profiler_mutex;
std::unordered_set<std::string> thread_instance::open_profilers;
#endif

// Global static unordered map of parent GUIDs to child GUIDs
// to handle "overlapping timer" problem.
std::unordered_map<uint64_t, std::vector<profiler*>* > thread_instance::children_to_resume;
static std::mutex _profiler_stack_mutex;

thread_instance& thread_instance::instance(bool is_worker) {
  if( _instance == nullptr ) {
    // first time called by this thread
    // construct test element to be used in all subsequent calls from this thread
    _instance = new thread_instance(is_worker);
    if (is_worker) {
        _instance->_id = _num_threads++;
    }
    _instance->_runtime_id = _instance->_id; // can be set later, if necessary
    _active_threads++;
  }
  return *_instance;
}

void thread_instance::delete_instance(void) {
  if (_instance != nullptr) {
    delete(_instance);
    _instance = nullptr;
  }
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
    // look up the address
    {
        read_lock_type l(_function_map_mutex);
        auto it = _function_map.find(function_address);
        if (it != _function_map.end()) {
          return (*it).second;
        } // else...
    }
#ifdef APEX_HAVE_BFD
    // resolve the address
    string * name = lookup_address(function_address, false);
    {
        write_lock_type l(_function_map_mutex);
        _function_map[function_address] = *name;
        delete(name);
    }
    read_lock_type l(_function_map_mutex);
    return _function_map[function_address];
#else
    stringstream ss;
    const char * progname = program_path();
    if (progname == NULL) {
        ss << "UNRESOLVED  ADDR 0x" << hex << function_address;
    } else {
        ss << "UNRESOLVED " << string(progname) << " ADDR " << hex << function_address;
    }
    string name = string(ss.str());
    {
        write_lock_type l(_function_map_mutex);
        _function_map[function_address] = name;
    }
    return name;
#endif
}

void thread_instance::set_current_profiler(profiler * the_profiler) {
    instance().current_profiler = the_profiler;
    instance().current_profilers.push_back(the_profiler);
    // restore the children here?
    std::unique_lock<std::mutex> l(_profiler_stack_mutex);
    auto tmp = children_to_resume.find(the_profiler->task_id->_guid);
    if (tmp != children_to_resume.end()) {
        auto myvec = tmp->second;
        for (profiler * myprof : *myvec) {
            resume(myprof);
            instance().current_profilers.push_back(myprof);
        }
        delete myvec;
        children_to_resume.erase(the_profiler->task_id->_guid);
    }
}

void thread_instance::clear_current_profiler(profiler * the_profiler) {
    __thread static bool fixing_stack = false;
    instance().current_profiler = nullptr;
    // this is a serious problem...
    if (instance().current_profilers.empty()) { 
        std::cerr << "Warning! empty profiler stack!\n";
        assert(false);
        return;
    }
    if (fixing_stack) {return;}
    auto tmp = instance().current_profilers.back();
    /* Uh-oh! Someone has caused the dreaded "overlapping timer" problem to
     * happen! No problem - stop the child timer.
     * Keep the children around, along with a reference to the parent's
     * guid so that if/when we see this parent again, we can restart
     * the children timers. */
    if (tmp != the_profiler) {
        fixing_stack = true;
        uint64_t guid = the_profiler->task_id->_guid;
        std::vector<profiler*> * children = new vector<profiler*>();
        while (tmp != the_profiler) {
            // if the guid isn't set, we can't support this runtime.
            assert(guid > 0);
            /* Make a copy of the profiler object on the top of the stack. */
            profiler * profiler_copy = new profiler(*tmp);
            children->push_back(tmp);
            /* Stop the copy. The original will get reset when the
            parent resumes. */
            stop(profiler_copy);  // we better be re-entrant safe!
            // pop the original child, we've saved it in the vector
            instance().current_profilers.pop_back();
            // this is a serious problem...
            if (instance().current_profilers.empty()) { 
                std::cerr << "Warning! empty profiler stack!\n";
                assert(false);
                return;
            }
            // get the new top of the stack
            tmp = instance().current_profilers.back();
        }
        fixing_stack = false;
        std::unique_lock<std::mutex> l(_profiler_stack_mutex);
        children_to_resume[guid] = children;
    }
    // pop this timer off the stack.
    instance().current_profilers.pop_back();
}

profiler * thread_instance::get_current_profiler(void) {
    return instance().current_profiler;
}

}
