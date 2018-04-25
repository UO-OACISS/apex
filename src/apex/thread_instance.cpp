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
// another method, just for the special Apple folks.
#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <assert.h>

using namespace std;

// the current thread's TASK index/count in APEX
APEX_NATIVE_TLS uint64_t my_task_id = 0;

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
      /* reverse the TID and shift it 32 bits, so we can use it to generate
         task-private GUIDS that are unique within the process space. */
        _instance->_id_reversed = ((uint64_t)(simple_reverse((uint32_t)_instance->_id))) << 32;
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
        if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
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
#if defined(__APPLE__)
    // resolve the address
    Dl_info info;
    int rc = dladdr((const void *)function_address, &info);
    if (rc != 0) {
        string name(info.dli_sname);
        {
            write_lock_type l(_function_map_mutex);
            _function_map[function_address] = name;
        }
        read_lock_type l(_function_map_mutex);
        return _function_map[function_address];
    }
#endif
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
    instance().current_profilers.push_back(the_profiler);
}

profiler * thread_instance::restore_children_profilers(task_wrapper * tt_ptr) {
    profiler * parent = instance().get_current_profiler();
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
        thread_instance::set_current_profiler((*myprof));
    }
    // clear the vector.
    myvec->clear();
    // The caller of this function wants the parent, not these leaves.
    return parent;
}

void thread_instance::clear_current_profiler(profiler * the_profiler, bool save_children, task_wrapper * tt_ptr) {
    // this is a stack variable that provides safety when using recursion.
    static APEX_NATIVE_TLS bool fixing_stack = false;
    // this is a serious problem...
    if (instance().current_profilers.empty()) { 
        // unless...we happen to be exiting.  Bets are off.
        if (apex_options::suspend() == true) { return; }
        std::cerr << "Warning! empty profiler stack!!!\n";
        assert(false);
        // redundant, but assert gets bypassed in a debug build.
        abort();
    }
    // check for recursion
    if (fixing_stack) {return;}
    // get the current stack of timers
    auto &the_stack = instance().current_profilers;
    auto tmp = the_stack.back();
    /* Uh-oh! Someone has caused the dreaded "overlapping timer" problem to
     * happen! No problem - stop the child timer.
     * Keep the children around, along with a reference to the parent's
     * guid so that if/when we see this parent again, we can restart
     * the children timers. */
    if (the_stack.size() > 1 && tmp != the_profiler) {
        fixing_stack = true;
        // if the data pointer location isn't available, we can't support this runtime.
        // create a vector to store the children
        if (save_children == true) {
            assert(tt_ptr != nullptr);
        }
        while (tmp != the_profiler) {
            if (save_children == true) {
                // if we are yielding, we need to stop the children
                /* Make a copy of the profiler object on the top of the stack. */
                profiler * profiler_copy = new profiler(*tmp);
                tt_ptr->data_ptr.push_back(tmp);
                /* Stop the copy. The original will get reset when the
                parent resumes. */
                stop(profiler_copy);  // we better be re-entrant safe!
            } else {
                // since we aren't yielding, just stop the children.
                stop(tmp);  // we better be re-entrant safe!
            }
            // pop the original child, we've saved it in the vector
            the_stack.pop_back();
            // this is a serious problem...
            if (the_stack.empty()) { 
                // unless...we happen to be exiting.  Bets are off.
                if (apex_options::suspend() == true) { return; }
                std::cerr << "Warning! empty profiler stack!\n";
                assert(false);
                abort();
                return;
            }
            // get the new top of the stack
            tmp = the_stack.back();
        }
        // done with the stack, allow proper recursion again.
        fixing_stack = false;
    }
    // pop this timer off the stack.
    the_stack.pop_back();
}

profiler * thread_instance::get_current_profiler(void) {
    if (instance().current_profilers.empty()) { return nullptr; }
    return instance().current_profilers.back();
}

}
