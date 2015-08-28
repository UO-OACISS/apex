#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include "pthread_wrapper.h"
#include <boost/atomic.hpp>
#include <iostream>
#include <new>
#include <system_error>


struct apex_system_wrapper_t
{
  bool initialized;
  apex_system_wrapper_t() : initialized(true) {
    apex::init("APEX Pthread Wrapper");
  }
  virtual ~apex_system_wrapper_t() {
    apex::finalize();
  }
};

bool initialize_apex_system(void) {
  // this static object will be created once, when we need it.
  // when it is destructed, the apex finalization will happen.
  static apex_system_wrapper_t initializer;
  if (initializer.initialized) {
	return false;
  }
  return true;
}


///////////////////////////////////////////////////////////////////////////////
// Below is the pthread_create wrapper
///////////////////////////////////////////////////////////////////////////////

static pthread_key_t wrapper_flags_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

void delete_key(void* wrapped) {
  if (!wrapped) {
    free(wrapped);
  }
}

static void make_key() { 
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

struct apex_pthread_pack
{
  start_routine_p start_routine;
  void * arg;
};

extern "C"
void * apex_pthread_function(void *arg)
{
  apex_pthread_pack * pack = (apex_pthread_pack*)arg;
  apex::profiler * p = nullptr;
  void * ret = nullptr;
  try {
    apex::register_thread("APEX pthread wrapper");
    p = apex::start((apex_function_address)pack->start_routine);
    ret = pack->start_routine(pack->arg);
    apex::stop(p);
    apex::exit_thread();
  } catch (std::bad_alloc& ba) {
    std::cerr << "bad_alloc caught: " << ba.what() << '\n';
  } catch(const std::system_error& e) {
    std::cout << "Caught system_error with code " << e.code() 
              << " meaning " << e.what() << '\n';
  }
  
  delete pack;
  return ret;
}

extern "C"
int apex_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr,
    start_routine_p start_routine, void * arg)
{
  (void) pthread_once(&key_once, make_key);
  int * wrapped = (int*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = (int*)malloc(sizeof(int));
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = 0;
  }

  int retval;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_create_call(threadp, attr, start_routine, arg);
  } else {
    *wrapped = 1;
    initialize_apex_system();
    apex_pthread_pack * pack = new apex_pthread_pack;
    pack->start_routine = start_routine;
    pack->arg = arg;

    retval = pthread_create_call(threadp, attr, apex_pthread_function, (void*)pack);
    *wrapped = 0;
  }
  return retval;
}

extern "C"
int apex_pthread_join_wrapper(pthread_join_p pthread_join_call,
    pthread_t thread, void ** retval)
{
  (void) pthread_once(&key_once, make_key);
  int * wrapped = (int*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = (int*)malloc(sizeof(int));
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = 0;
  }

  int ret;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    ret = pthread_join_call(thread, retval);
  } else {
    *wrapped = 1;
    ret = pthread_join_call(thread, retval);
    *wrapped = 0;
  }
  return ret;
}

extern "C"
void apex_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr)
{
  (void) pthread_once(&key_once, make_key);
  int * wrapped = (int*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = (int*)malloc(sizeof(int));
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = 0;
  }

  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    pthread_exit_call(value_ptr);
  } else {
    *wrapped = 1;
    apex::exit_thread();
    pthread_exit_call(value_ptr);
    *wrapped = 0;
  }
}

#ifdef APEX_PTHREAD_BARRIER_AVAILABLE
extern "C"
int apex_pthread_barrier_wait_wrapper(pthread_barrier_wait_p pthread_barrier_wait_call,
    pthread_barrier_t * barrier)
{
  (void) pthread_once(&key_once, make_key);
  int * wrapped = (int*)pthread_getspecific(wrapper_flags_key);
  if (!wrapped) {
    wrapped = (int*)malloc(sizeof(int));
    pthread_setspecific(wrapper_flags_key, (void*)wrapped);
    *wrapped = 0;
  }

  int retval;
  if(*wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_barrier_wait_call(barrier);
  } else {
    *wrapped = 1;
    apex::profiler * p = apex::start("pthread_barrier_wait");
    retval = pthread_barrier_wait_call(barrier);
    apex::stop(p);
    *wrapped = 0;
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
