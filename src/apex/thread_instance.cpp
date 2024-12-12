/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex_api.hpp" // make this the first include.
#include "apex.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "apex_assert.h"

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

#include "address_resolution.hpp"

using namespace std;

// the current thread's TASK index/count in APEX
APEX_NATIVE_TLS uint64_t my_task_id = 0;

namespace apex {

/*
// Global static pointer used to ensure a single instance of the class.
APEX_NATIVE_TLS thread_instance * thread_instance::_instance(nullptr);
// Global static count of all known threads in system
std::atomic_int thread_instance::_num_threads(0);
// Global static count of worker threads in system
std::atomic_int thread_instance::_num_workers(0);
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
*/
#ifdef APEX_DEBUG
// Global static mutex to control access for debugging purposes
std::mutex thread_instance::_open_profiler_mutex;
std::unordered_set<std::string> thread_instance::open_profilers;
#endif

thread_instance& thread_instance::instance(bool is_worker) {
  // thread specific data
  static APEX_NATIVE_TLS thread_instance _instance(is_worker);
  return _instance;
}

thread_instance::~thread_instance(void) {
    if(apex_options::use_verbose()) {
        if(_is_worker) {
            std::cout << "worker ";
        } else {
            std::cout << "non-worker ";
        }
        std::cout << "thread " << _id << " exiting... " << __APEX_FUNCTION__ << std::endl;
    }
    if (get_program_over()) { return; }
    if (_top_level_timer != nullptr) {
        stop(_top_level_timer);
        _top_level_timer = nullptr;
    }
    if (_id == 0) {
        finalize();
    } else {
#if defined(APEX_HAVE_HPX)
        if(_is_worker) {
            finalize();
        }
#endif
    }
    common()._active_threads--;
}

void thread_instance::set_worker(bool is_worker) {
  // if was previously not a worker...
  // ...and is now a worker...
  if (!instance()._is_worker && is_worker) {
      instance()._id = common()._num_workers++;
  } else if (instance()._is_worker && !is_worker) {
      instance()._id = common()._num_workers--;
  }
  instance()._is_worker = is_worker;
  std::unique_lock<std::mutex> l(common()._worker_map_mutex);
  common()._worker_map.insert(std::pair<int,bool>(get_id(), is_worker));
}

string thread_instance::get_name(void) {
  return instance()._top_level_timer_name;
}

void thread_instance::set_name(string name) {
  if (instance()._top_level_timer_name.empty())
  {
    instance()._top_level_timer_name = name;
    {
        std::unique_lock<std::mutex> l(common()._name_map_mutex);
        common()._name_map[name] = get_id();
    }
#if defined(APEX_HAVE_HPX)
    if (name.find("worker-thread") != string::npos) {
      instance().set_worker(true);
    }
#else
    instance().set_worker(true);
#endif
  }
}

int thread_instance::map_name_to_id(string name) {
  //cout << "Looking for " << name << endl;
  int tmp = -1;
  {
    std::unique_lock<std::mutex> l(common()._name_map_mutex);
    map<string, int>::const_iterator it = common()._name_map.find(name);
    if (it != common()._name_map.end()) {
      tmp = (*it).second;
    }
  }
  return tmp;
}

bool thread_instance::map_id_to_worker(int id) {
  //cout << "Looking for " << name << endl;
  bool worker = false;
  {
    std::unique_lock<std::mutex> l(common()._worker_map_mutex);
    map<int, bool>::const_iterator it = common()._worker_map.find(id);
    if (it != common()._worker_map.end()) {
      worker = (*it).second;
    }
  }
  return worker;
}

const char* thread_instance::program_path(void) {

#if defined(_WIN32) || defined(_WIN64)

    if (common()._program_path == nullptr) {
        char path[MAX_PATH + 1] = { '\0' };
        if (!GetModuleFileName(nullptr, path, sizeof(path)))
            return nullptr;
        common()._program_path = new string(path);
    }
    return common()._program_path->c_str();

#elif defined(__linux) || defined(linux) || defined(__linux__)

    if (common()._program_path == nullptr) {
        char path[PATH_MAX];
        memset(path,0,PATH_MAX);
        if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
            return nullptr;
        }
        common()._program_path = new string(path);
    }
    return common()._program_path->c_str();

#elif defined(__APPLE__)

    if (common()._program_path == nullptr) {
        char path[PATH_MAX + 1];
        std::uint32_t len = sizeof(path) / sizeof(path[0]);

        if (0 != _NSGetExecutablePath(path, &len))
            return nullptr;

        path[len] = '\0';
        common()._program_path = new string(path);
    }
    return common()._program_path->c_str();

#elif defined(__FreeBSD__)

    if (common()._program_path == nullptr) {
        int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
        size_t cb = 0;
        sysctl(mib, 4, nullptr, &cb, nullptr, 0);
        if (!cb)
            return nullptr;

        std::vector<char> buf(cb);
        sysctl(mib, 4, &buf[0], &cb, nullptr, 0);
        common()._program_path = new string(&buf[0]);
    }
    return common()._program_path->c_str();

#else
#  error Unsupported platform
    return nullptr;
#endif
}

string thread_instance::map_addr_to_name(apex_function_address function_address) {
    // look up the address
    {
        read_lock_type l(common()._function_map_mutex);
        auto it = _function_map.find(function_address);
        if (it != _function_map.end()) {
          return (*it).second;
        } // else...
    }
    // resolve the address
    string * name = lookup_address(function_address, false);
    {
        write_lock_type l(common()._function_map_mutex);
        _function_map[function_address] = *name;
        delete(name);
    }
    read_lock_type l(common()._function_map_mutex);
    return _function_map[function_address];
}

profiler* thread_instance::restore_children_profilers(
    std::shared_ptr<task_wrapper> &tt_ptr) {
    profiler* parent = instance().get_current_profiler();
    // if there are no children to restore, return.
    if (tt_ptr == nullptr || tt_ptr->data_ptr.size() == 0) {return parent;}
    // Get the vector of children that we stored
    std::vector<profiler*> * myvec = &tt_ptr->data_ptr;
    // iterate over the children, in reverse order.
    for (std::vector<profiler*>::reverse_iterator myprof = myvec->rbegin();
         myprof != myvec->rend(); ++myprof) {
        resume((*myprof));
        // make sure to set the current profiler - the profiler_listener
        // is bypassed by the resume method, above.  It's the listener that
        // sets the current profiler when a timer is started
        //thread_instance::instance().set_current_profiler((*myprof));
    }
    // clear the vector.
    myvec->clear();
    // The caller of this function wants the parent, not these leaves.
    return parent;
}

void thread_instance::clear_current_profiler(
    bool save_children, std::shared_ptr<task_wrapper> &tt_ptr) {
    static APEX_NATIVE_TLS bool fixing_stack = false;
    // check for recursion
    if (fixing_stack) {return;}
    // get the current profiler
    profiler* tmp = instance().current_profiler;
    // get the task wrapper's profiler
    profiler* the_profiler = tt_ptr->prof;
    // This thread has no running timers, do nothing.
    if (tmp == nullptr) {
        //printf("Setting current profiler to nullptr\n");
        return;
    }
    // if this profiler was started somewhere else, do nothing.
    if (the_profiler->thread_id != instance().get_id()) {
        //printf("Doing nothing with current profiler\n");
        return;
    }
    /* if we are doing a "stop", then just clear ourselves IF we are current. */
    if (!save_children) {
        if (tmp == the_profiler) {
            instance().current_profiler = tmp->untied_parent;
        }
        // otherwise do nothing.
        return;
    }
    // if the current profiler isn't this profiler, is it in the "stack"?
    // we know the current profiler and the one we are stopping are
    // on the same thread. Assume we are handling a "direct action" that was
    // yielded.
    if (save_children && (tmp != the_profiler)) {
        fixing_stack = true;
        // if the data pointer location isn't available, we can't support this runtime.
        // create a vector to store the children
        APEX_ASSERT(tt_ptr != nullptr);
        while (tmp != the_profiler) {
            // if we are yielding, we need to stop the children
            /* Make a copy of the profiler object on the top of the stack. */
            profiler * profiler_copy = new profiler(*tmp);
            tt_ptr->data_ptr.push_back(tmp);
            /* yield the copy. The original will get reset when the
            parent resumes. */
            yield(profiler_copy);  // we better be re-entrant safe!
            // this is a serious problem...or is it? no!
            if (tmp->untied_parent == nullptr) {
                instance().current_profiler = nullptr;
                return;
            }
            // get the new top of the stack
            tmp = tmp->untied_parent;
        }
        // done with the stack, allow proper recursion again.
        fixing_stack = false;
    }
    instance().current_profiler = tmp->untied_parent;
}

}
