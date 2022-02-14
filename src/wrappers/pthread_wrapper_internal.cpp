/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include "pthread_wrapper.h"
#include <iostream>
#include <new>
#include <system_error>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <memory>
#include <atomic>

std::atomic<int64_t> task_id(-1);

/*
 * This "class" is used to make sure APEX is initialized
 * before the first pthread is created and finalized
 * when this object is destroyed.
 */
struct apex_system_wrapper_t
{
  bool initialized;
  apex_system_wrapper_t() : initialized(true) {
    apex::init("APEX Pthread Wrapper",0,1);
    /*
     * Here we are limiting the stack size to whatever the user requests.
     * Do it after we initialized APEX, because APEX spawns two other threads
     * that may require more.
     */
    struct rlimit limits;
    getrlimit(RLIMIT_STACK,&limits);
    if (apex::apex_options::pthread_wrapper_stack_size() != 0) {
      limits.rlim_cur = apex::apex_options::pthread_wrapper_stack_size();
      limits.rlim_max = apex::apex_options::pthread_wrapper_stack_size();
      /*
    } else {
      limits.rlim_cur = 16384;
      limits.rlim_max = 16384;
      */
    }
    int rc = setrlimit(RLIMIT_STACK,&limits);
    if (rc != 0) {
      std::cerr << "WARNING: unable to cap the stack size..." << std::endl;
    }
  }
  virtual ~apex_system_wrapper_t() {
    apex::finalize();
  }
};

/*
 * The "static" method to initialize APEX
 */
bool initialize_apex_system(void) {
  // this static object will be created once, when we need it.
  // when it is destructed, the apex finalization will happen.
  static apex_system_wrapper_t initializer;
  if (initializer.initialized) {
    return false;
  }
  return true;
}

/*
 * Here are the keys for thread local variables
 */
static pthread_key_t wrapper_flags_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

/*
 * This thread-local wrapper object is used to register, start, pause,
 * restart, stop, and exit the timer around the pthread object.
 */
class apex_wrapper {
public:
  apex_wrapper(void) : _wrapped(false), _twp(nullptr), _timing(false) {
  }
  apex_wrapper(start_routine_p func, std::shared_ptr<apex::task_wrapper> parent) :
    _wrapped(false), _twp(nullptr), _timing(false) {
    apex::register_thread("APEX pthread wrapper", parent);
    //_twp = apex::new_task((apex_function_address)func, UINTMAX_MAX, parent);
    _twp = apex::new_task((apex_function_address)func);
  }
  ~apex_wrapper() {
    this->stop();
    apex::exit_thread();
  }
  void start(void) {
    if (_twp != nullptr) {
      apex::start(_twp);
      _timing = true;
    }
  }
  void restart(void) {
    if (_twp != nullptr) {
      apex::start(_twp);
      _timing = true;
    }
  }
  void yield(void) {
    if (_timing) {
      apex::yield(_twp);
      _timing = false;
    }
  }
  void stop(void) {
    if (_timing) {
      apex::stop(_twp);
      _timing = false;
    }
  }
  bool _wrapped;
private:
  std::shared_ptr<apex::task_wrapper> _twp;
  bool _timing;
};

/*
 * This is the destructor used to delete the thread local variable
 */
void delete_key(void* wrapper) {
  if (wrapper != NULL) {
    apex_wrapper * tmp = (apex_wrapper*)wrapper;
    delete tmp;
  }
}

/*
 * This key is made once, by the pthread system.
 */
static void make_key() {
  // create the key
  int rc = pthread_key_create(&wrapper_flags_key, &delete_key);
  switch (rc) {
    case EAGAIN:
      std::cout << "ERROR: The system lacked the necessary resources to create another thread-specific data key, or the system-imposed limit on the total number of keys per process {PTHREAD_KEYS_MAX} has been exceeded." << std::endl;
      break;
    case ENOMEM:
      std::cout << "ERROR: Insufficient memory exists to create the key." << std::endl;
      break;
    default:
      break;
  }
}

/*
 * this structure is used to wrap the true pthread function and its argument
 */
struct apex_pthread_pack
{
  start_routine_p start_routine;
  void * arg;
  std::shared_ptr<apex::task_wrapper> parent;
};

/*
 * This method is a proxy around the true pthread function.
 */
extern "C"
void * apex_pthread_function(void *arg)
{
  apex_pthread_pack * pack = (apex_pthread_pack*)arg;
  void * ret = nullptr;

  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  // if it doesn't exist, create one.
  if (wrapper == NULL) {
    wrapper = new apex_wrapper(pack->start_routine, pack->parent);
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }

  // start the timer
  wrapper->start();
  // call the *real* function
  ret = pack->start_routine(pack->arg);
  // stop the timer
  wrapper->stop();
  // delete our packer
  delete pack;
  return ret;
}

/*
 * Helper method to get or create the thread local variable.
 */
apex_wrapper * get_tl_wrapper (void) {
  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  // if it doesn't exist, create one.
  if (wrapper == NULL) {
    wrapper = new apex_wrapper();
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }
  return wrapper;
}

///////////////////////////////////////////////////////////////////////////////
// Below is the pthread_create wrapper
///////////////////////////////////////////////////////////////////////////////

extern "C"
int apex_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr,
    start_routine_p start_routine, void * arg)
{
  // disable the memory wrapper
  apex::in_apex prevent_problems;
  std::shared_ptr<apex::task_wrapper> parent_task = apex::new_task("pthread_create");
  apex::start(parent_task);
  // JUST ONCE, create the key
  (void) pthread_once(&key_once, make_key);
  // get the thread-local variable
  apex_wrapper * wrapper = get_tl_wrapper();
  int retval;
  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_create_call(threadp, attr, start_routine, arg);
  } else {
    wrapper->_wrapped = true;
    // JUST ONCE, initialize APEX. This can't be done in the make_key call,
    // because apex::init will spawn 2 or more threads, which causes deadlock
    initialize_apex_system();

    // pack the real start_routine and argument
    apex_pthread_pack * pack = new apex_pthread_pack;
    pack->start_routine = start_routine;
    pack->arg = arg;
    pack->parent = parent_task;

    // create the pthread, pass in our proxy function and the packed data
    retval = pthread_create_call(threadp, attr, apex_pthread_function, (void*)pack);

    // register the dependency with APEX.
    /*
    if (retval == 0) {
        apex::new_task((apex_function_address)start_routine, ++task_id);
    }
    */
    wrapper->_wrapped = false;
  }
  apex::stop(parent_task);
  return retval;
}

extern "C"
int apex_pthread_join_wrapper(pthread_join_p pthread_join_call,
    pthread_t thread, void ** retval)
{
  // disable the memory wrapper
  apex::in_apex prevent_problems;
  apex_wrapper * wrapper = get_tl_wrapper();

  int ret;
  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    ret = pthread_join_call(thread, retval);
  } else {
    wrapper->_wrapped = true;
    /* DON'T do this for now - it creates too many events. Until we can figure
     * out how to get the profiler_listener to process the queues faster... */
    // stop our current timer
    wrapper->yield();
    // start a new timer for the join event
    apex::profiler * p = apex::start("pthread_join");
    ret = pthread_join_call(thread, retval);
    // stop the timer for the join
    apex::stop(p);
    // restart our timer around the parent task
    wrapper->restart();
    wrapper->_wrapped = false;
  }
  return ret;
}

#if 0
extern "C"
void apex_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr)
{
  //(void) pthread_once(&key_once, make_key);
  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  if (wrapper == NULL) {
    wrapper = new apex_wrapper();
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
    wrapper->_wrapped = 0;
  }

  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    pthread_exit_call(value_ptr);
  } else {
    wrapper->_wrapped = 1;
    //apex::exit_thread();
    pthread_exit_call(value_ptr);
    wrapper->_wrapped = 0;
  }
}

#ifdef APEX_PTHREAD_BARRIER_AVAILABLE
extern "C"
int apex_pthread_barrier_wait_wrapper(pthread_barrier_wait_p pthread_barrier_wait_call,
    pthread_barrier_t * barrier)
{
  //(void) pthread_once(&key_once, make_key);
  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  if (wrapper == NULL) {
    wrapper = new apex_wrapper();
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }

  int retval;
  wrapper->yield();
  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_barrier_wait_call(barrier);
  } else {
    wrapper->_wrapped = 1;
    apex::profiler * p = apex::start("pthread_barrier_wait");
    retval = pthread_barrier_wait_call(barrier);
    apex::stop(p);
    wrapper->_wrapped = 0;
  }
  wrapper->restart();
  return retval;
}
#endif /* APEX_PTHREAD_BARRIER_AVAILABLE */
#endif

extern "C"
int apex_pthread_create(pthread_t * threadp, const pthread_attr_t * attr,
    start_routine_p start_routine, void * arg)
{
  return apex_pthread_create_wrapper(pthread_create, threadp, attr, start_routine, arg);
}

extern "C"
int apex_pthread_join(pthread_t thread, void ** retval)
{
  return apex_pthread_join_wrapper(pthread_join, thread, retval);
}

#if 0
extern "C"
void apex_pthread_exit(void * value_ptr)
{
  apex_pthread_exit_wrapper(pthread_exit, value_ptr);
}

#ifdef APEX_PTHREAD_BARRIER_AVAILABLE
extern "C"
int apex_pthread_barrier_wait(pthread_barrier_t * barrier)
{
  return apex_pthread_barrier_wait_wrapper(pthread_barrier_wait, barrier);
}

#endif /* APEX_PTHREAD_BARRIER_AVAILABLE */
#endif
