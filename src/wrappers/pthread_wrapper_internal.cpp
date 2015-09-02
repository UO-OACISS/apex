#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include "pthread_wrapper.h"
#include <boost/atomic.hpp>
#include <iostream>
#include <new>
#include <system_error>
#include <unistd.h>
#include <limits.h>

struct apex_system_wrapper_t
{
  bool initialized;
  apex_system_wrapper_t() : initialized(true) {
    apex::init("APEX Pthread Wrapper");
  }
  virtual ~apex_system_wrapper_t() {
    apex::apex_options::use_screen_output(true);
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

class apex_wrapper {
public:
  apex_wrapper(void) : _wrapped(false), _p(nullptr), _stopped(false), _func(nullptr) {
  }
  apex_wrapper(start_routine_p func) : _wrapped(false), _p(nullptr), _stopped(false), _func(func) {
    apex::register_thread("APEX pthread wrapper");
  }
  ~apex_wrapper() {
    this->stop();
    apex::exit_thread();
  }
  void start(void) {
    if (_func != NULL) {
      _p = apex::start((apex_function_address)_func);
      _stopped = false;
    }
  }
  void restart(void) {
    if (_func != NULL) {
      _p = apex::resume((apex_function_address)_func);
      _stopped = false;
    }
  }
  void yield(void) {
    if (!_stopped && _p != NULL && !_p->stopped) {
      apex::stop(_p);
      _stopped = true;
      _p = nullptr;
    }
  }
  void stop(void) {
    if (!_stopped && _p != NULL && !_p->stopped) {
      apex::stop(_p);
      _stopped = true;
      _p = nullptr;
    }
  }
  bool _wrapped;
private:
  apex::profiler * _p;
  bool _stopped;
  start_routine_p _func;
};

void delete_key(void* wrapper) {
  if (wrapper != NULL) {
    apex_wrapper * tmp = (apex_wrapper*)wrapper;
    delete tmp;
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
  void * ret = nullptr;

  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  if (wrapper == NULL) {
    wrapper = new apex_wrapper(pack->start_routine);
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }

  wrapper->start();
  ret = pack->start_routine(pack->arg);
  wrapper->stop();
  delete pack;
  return ret;
}

extern "C"
int apex_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr,
    start_routine_p start_routine, void * arg)
{
  (void) pthread_once(&key_once, make_key);
  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  if (wrapper == NULL) {
    wrapper = new apex_wrapper();
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }

  int retval;
  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    retval = pthread_create_call(threadp, attr, start_routine, arg);
  } else {
    wrapper->_wrapped = 1;
    initialize_apex_system();
    apex_pthread_pack * pack = new apex_pthread_pack;
    pack->start_routine = start_routine;
    pack->arg = arg;

    pthread_attr_t tmp;
    bool destroy_attr = false;
    if (attr == NULL) {
      destroy_attr = true;
      pthread_attr_init(&tmp);
      attr = &tmp;
    }
	size_t defaultSize = 0L;
    if (pthread_attr_getstacksize(attr, &defaultSize)) {
      std::cerr << "APEX: ERROR - failed to get default pthread stack size.\n";
    }
	if (defaultSize > PTHREAD_STACK_MIN) {
      if(pthread_attr_setstacksize(const_cast<pthread_attr_t*>(attr), PTHREAD_STACK_MIN)) {
        std::cerr << "APEX: ERROR - failed to change pthread stack size from " << defaultSize << " to " << PTHREAD_STACK_MIN << std::endl;
      }
    }

    retval = pthread_create_call(threadp, attr, apex_pthread_function, (void*)pack);
	if (destroy_attr) {
	  pthread_attr_destroy(&tmp);
	}
    wrapper->_wrapped = 0;
  }
  return retval;
}

extern "C"
int apex_pthread_join_wrapper(pthread_join_p pthread_join_call,
    pthread_t thread, void ** retval)
{
  (void) pthread_once(&key_once, make_key);
  apex_wrapper * wrapper = (apex_wrapper*)pthread_getspecific(wrapper_flags_key);
  if (wrapper == NULL) {
    wrapper = new apex_wrapper();
    pthread_setspecific(wrapper_flags_key, (void*)wrapper);
  }

  int ret;
  wrapper->yield();
  if(wrapper->_wrapped) {
    // Another wrapper has already intercepted the call so just pass through
    ret = pthread_join_call(thread, retval);
  } else {
    wrapper->_wrapped = 1;
    apex::profiler * p = apex::start("pthread_join");
    ret = pthread_join_call(thread, retval);
    apex::stop(p);
    wrapper->_wrapped = 0;
  }
  wrapper->restart();
  return ret;
}

extern "C"
void apex_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr)
{
  (void) pthread_once(&key_once, make_key);
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
  (void) pthread_once(&key_once, make_key);
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
