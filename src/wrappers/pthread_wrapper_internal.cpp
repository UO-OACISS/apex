#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include "pthread_wrapper.h"
#include <boost/atomic.hpp>
#include <iostream>

// constructor and destructor

/*void __attribute__ ((constructor)) premain()
{
	apex::init("APEX Pthread Wrapper");
}*/

void __attribute__ ((destructor)) postmain()
{
	apex::finalize();
}

///////////////////////////////////////////////////////////////////////////////
// Below is the pthread_create wrapper
///////////////////////////////////////////////////////////////////////////////

pthread_key_t wrapper_flags_key;

struct apex_pthread_pack
{
  start_routine_p start_routine;
  void * arg;
};

extern "C"
void * apex_pthread_function(void *arg)
{
  apex_pthread_pack * pack = (apex_pthread_pack*)arg;

  apex::register_thread("pthread");
  apex::profiler * p = apex::start((apex_function_address)pack->start_routine);
  void * ret = pack->start_routine(pack->arg);
#ifndef APEX_TBB_SUPPORT
  // Thread 0 in TBB will not wait for the other threads to finish
  // (it does not join). DO NOT stop the timer for this thread, but
  // let thread 0 do that when it shuts down all threads on exit.
  apex::stop(p);
#endif
  delete pack;
  return ret;
}

bool initialize(void) {
  static boost::atomic<bool> initialized(false);
  if (!initialized) {
	apex::init("APEX Pthread Wrapper");
	initialized = true;
  }
  return true;
}

extern "C"
int apex_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr,
    start_routine_p start_routine, void * arg)
{
  initialize();
  bool * wrapped = (bool*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = new bool;
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = false;
  }

  int retval;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_create_call(threadp, attr, start_routine, arg);
  } else {
    *wrapped = true;
    apex_pthread_pack * pack = new apex_pthread_pack;
    pack->start_routine = start_routine;
    pack->arg = arg;

    retval = pthread_create_call(threadp, attr, apex_pthread_function, (void*)pack);
    *wrapped = false;
  }
  return retval;
}

extern "C"
int apex_pthread_join_wrapper(pthread_join_p pthread_join_call,
    pthread_t thread, void ** retval)
{
  bool * wrapped = (bool*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = new bool;
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = false;
  }

  int ret;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    ret = pthread_join_call(thread, retval);
  } else {
    *wrapped = true;
    ret = pthread_join_call(thread, retval);
    *wrapped = false;
  }
  return ret;
}

extern "C"
void apex_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr)
{
  bool * wrapped = (bool*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = new bool;
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = false;
  }

  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    pthread_exit_call(value_ptr);
  } else {
    *wrapped = true;
    apex::exit_thread();
    pthread_exit_call(value_ptr);
    *wrapped = false;
  }
}

#ifdef APEX_PTHREAD_BARRIER_AVAILABLE
extern "C"
int apex_pthread_barrier_wait_wrapper(pthread_barrier_wait_p pthread_barrier_wait_call,
    pthread_barrier_t * barrier)
{
  bool * wrapped = (bool*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = new bool;
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = false;
  }

  int retval;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_barrier_wait_call(barrier);
  } else {
    *wrapped = true;
    apex::profiler * p = apex::start("pthread_barrier_wait");
    retval = pthread_barrier_wait_call(barrier);
    apex::stop(p);
    *wrapped = false;
  }
  return retval;
}
#endif /* APEX_PTHREAD_BARRIER_AVAILABLE */

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
