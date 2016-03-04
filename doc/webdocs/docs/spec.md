# APEX Specification *(DRAFT)*

*...to be fully implemented in v0.2 release.*

## READ ME FIRST!

The API specification is provided for users who wish to instrument their
own applications, or who wish to instrument a runtime. Please note that 
the [HPX](usage.md#hpx-louisiana-state-university),
[HPX-5](usage.md#hpx-5-indiana-university) and
[OpenMP](usecases.md#openmp-example) (using the LLVM OpenMP implementation
with draft OMPT support) runtimes have already been instrumented,
and that users typically do not have to make any calls to the APEX API, other
than to add application level timers or to write custom policy rules.

## Introduction

This page contains the API specification for APEX. The API specification
provides a high-level overview of the API and its functionality. The
implementation has Doxygen comments inserted, so for full implementation
details, please see the [API Reference Manual](refman.md).  

### A note about C++

The following specification contains both the C and the the C++ API. Typically,
the C++ names use overloading for different argument lists, and will replace
the `apex_` prefix with the `apex::` namespace. Because both APIs return 
handles to internal APEX objects, the type definitions of these objects 
use the C naming convention.

### Terminology

Unfortunately, many terms in Computer Science are overloaded. The following definitions are in use in this document:

**Thread**: an operating system (OS) thread of execution. For example, Posix threads (pthreads).  

**Task**: a scheduled unit of work, such as an OpenMP task or an HPX thread. APEX timers are typically used to measure tasks.

### C example

The following is a very small C program that uses the APEX API. For more
examples, please see the programs in the `src/examples` and `src/unit_tests/C`
directories of the APEX source code.

``` c
#include <unistd.h>
#include <stdio.h>
#include "apex.h"

int foo(int i) {
    /* start an APEX timer for the function foo */
    apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &foo);
    int j = i * i;
    /* stop the APEX timer */
    apex_stop(profiler);
    return j;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex_init("apex_start unit test");
    /* start a timer, passing in the address of the main function */
    apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
    int i,j = 0;
    for (i = 0 ; i < 3 ; i++) {
        j += foo(i);
    }
    /* stop the timer */
    apex_stop(profiler);
    /* finalize APEX */
    apex_finalize();
    /* free all memory allocated by APEX */
    apex_cleanup();
    return 0;
}
```

### C++ example

The following is a slightly more complicated C++ pthread program that uses the
APEX API. For more examples, please see the programs in the `src/examples` and
`src/unit_tests/C++` directories of the APEX source code.

``` c++
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include "apex_api.hpp"

void* someThread(void* tmp)
{
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker thread %d", *tid);
    /* Register this thread with APEX */
    apex::register_thread(name);
    /* Start a timer */
    apex::profiler* p = apex::start((apex_function_address)&someThread);
    /* ... */
    /* do some computation */
    /* ... */
    /* stop the timer */
    apex::stop(p);
    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test");
    /* set our node ID */
    apex::set_node_id(0);
    /* start a timer */
    apex::profiler* p = apex::start("main");
    /* Spawn two threads */
    pthread_t thread[2];
    int tid = 0;
    pthread_create(&(thread[0]), NULL, someThread, &tid);
    int tid2 = 1;
    pthread_create(&(thread[1]), NULL, someThread, &tid2);
    /* wait for the threads to finish */
    pthread_join(thread[0], NULL);
    pthread_join(thread[1], NULL);
    /* stop our main timer */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
    /* free all memory allocated by APEX */
    apex::cleanup();
    return 0;
}
```

## Constants, types and enumerations

### Constants

``` c
/** A null pointer representing an APEX profiler handle.
 * Used when a null APEX profile handle is to be passed in to
 * apex::stop when the profiler object was not retained locally.
 */
#define APEX_NULL_PROFILER_HANDLE (apex_profiler_handle)(NULL) // for comparisons

#define APEX_MAX_EVENTS 128 /*!< The maximum number of event types. Allows for ~20 custom events. */

#define APEX_NULL_FUNCTION_ADDRESS 0L // for comparisons
```

### Pre-defined types

``` c
/** The address of a C++ object in APEX.
 * Not useful for the caller that gets it back, but required
 * for stopping the timer later.
 */
typedef uintptr_t apex_profiler_handle; // address of internal C++ object

/** Not useful for the caller that gets it back, but required
 * for deregistering policies after registration.
 */
typedef uintptr_t apex_policy_handle; // address of internal C++ object

/** Rather than use void pointers everywhere, be explicit about
 * what the functions are expecting.
 */
typedef uintptr_t apex_function_address; // generic function pointer
```

### Enumerations

``` c
/**
 * Typedef for enumerating the different timer types
 */
typedef enum _apex_profiler_type {
    APEX_FUNCTION_ADDRESS = 0, /*!< The ID is a function (or instruction) address */
    APEX_NAME_STRING,          /*!< The ID is a character string */
    APEX_FUNCTOR               /*!< C++ Object with the () operator defined */
} apex_profiler_type;

/**
 * Typedef for enumerating the different event types
 */
typedef enum _event_type {
  APEX_INVALID_EVENT = -1,
  APEX_STARTUP = 0,        /*!< APEX is initialized */
  APEX_SHUTDOWN,       /*!< APEX is terminated */
  APEX_NEW_NODE,       /*!< APEX has registered a new process ID */
  APEX_NEW_THREAD,     /*!< APEX has registered a new OS thread */
  APEX_EXIT_THREAD,    /*!< APEX has exited an OS thread */
  APEX_START_EVENT,    /*!< APEX has processed a timer start event */
  APEX_RESUME_EVENT,   /*!< APEX has processed a timer resume event (the number
                           of calls is not incremented) */
  APEX_STOP_EVENT,     /*!< APEX has processed a timer stop event */
  APEX_YIELD_EVENT,    /*!< APEX has processed a timer yield event */
  APEX_SAMPLE_VALUE,   /*!< APEX has processed a sampled value */
  APEX_PERIODIC,       /*!< APEX has processed a periodic timer */
  APEX_CUSTOM_EVENT_1,   /*!< APEX has processed a custom event - useful for large
                           granularity application control events */
  APEX_CUSTOM_EVENT_2, // these are just here for padding, and so we can
  APEX_CUSTOM_EVENT_3, // test with them.
  APEX_CUSTOM_EVENT_4,
  APEX_CUSTOM_EVENT_5,
  APEX_CUSTOM_EVENT_6,
  APEX_CUSTOM_EVENT_7,
  APEX_CUSTOM_EVENT_8,
  APEX_UNUSED_EVENT = APEX_MAX_EVENTS // can't have more custom events than this
} apex_event_type;

/** 
 * Typedef for enumerating the OS thread states. 
 */
typedef enum _thread_state {
    APEX_IDLE,          /*!< Thread is idle */
    APEX_BUSY,          /*!< Thread is working */
    APEX_THROTTLED,     /*!< Thread is throttled (sleeping) */
    APEX_WAITING,       /*!< Thread is waiting for a resource */
    APEX_BLOCKED        /*!< Thread is otherwise blocked */
} apex_thread_state;

/**
 * Typedef for enumerating the different optimization strategies
 * for throttling.
 */
typedef enum {APEX_MAXIMIZE_THROUGHPUT,   /*!< maximize the number of calls to a
                                              timer/counter */
              APEX_MAXIMIZE_ACCUMULATED,  /*!< maximize the accumulated value of
                                              a timer/counter */
              APEX_MINIMIZE_ACCUMULATED   /*!< minimize the accumulated value of
                                              a timer/counter */
} apex_optimization_criteria_t;

/**
 * Typedef for enumerating the different optimization methods
 * for throttling.
 */
typedef enum {APEX_SIMPLE_HYSTERESIS,      /*!< optimize using sliding window of
                                               historical observations. A running
                                               average of the most recent N observations
                                               are used as the measurement. */
              APEX_DISCRETE_HILL_CLIMBING, /*!< Use a discrete hill climbing algorithm
                                               for optimization */
              APEX_ACTIVE_HARMONY          /*!< Use Active Harmony for optimization. */
} apex_optimization_method_t;

/** The type of a profiler object
 * 
 */
typedef enum _profile_type {
  APEX_TIMER,        /*!< This profile is a instrumented timer */
  APEX_COUNTER       /*!< This profile is a sampled counter */
} apex_profile_type;


```

## Data structures and classes

``` c
/** 
 * The APEX context when an event occurs. This context will be passed to 
 * any policies registered for this event.
 */
typedef struct _context {
    apex_event_type event_type;        /*!< The type of the event currently
                                           processing */
    apex_policy_handle* policy_handle; /*!< The policy handle for the current
                                           policy function */
    void * data;  /*!< Data associated with the event, such as the custom_data
                       for a custom_event */
} apex_context;

/**
 * The profile object for a timer in APEX. 
 * Returned by the apex_get_profile() call.
 */
typedef struct _profile {
    double calls;         /*!< Number of times a timer was called, or the number
                              of samples collected for a counter */
    double accumulated;   /*!< Accumulated values for all calls/samples */
    double sum_squares;   /*!< Running sum of squares calculation for all
                              calls/samples */
    double minimum;       /*!< Minimum value seen by the timer or counter */
    double maximum;       /*!< Maximum value seen by the timer or counter */
    apex_profile_type type; /*!< Whether this is a timer or a counter */
    double papi_metrics[8];  /*!< Array of accumulated PAPI hardware metrics */
} apex_profile;

/**
 * The APEX tuning request structures.
 */

typedef struct _apex_param {
    char * init_value;          /*!< Initial value */
    const char * value;         /*!< Current value */
    int num_possible_values;    /*!< Number of possible values */
    char * possible_values[];
} apex_param_struct;

typedef struct _apex_tuning_request {
    char * name;                                      /*!< Tuning request name */
    double (*metric)(void);                           /*!< function to return the address of the output parameter */
    int num_params;                                   /*!< number of tuning input parameters */
    char * param_names[];                             /*!< the input parameter names */
    apex_param_struct * params[];                     /*!< the input parameters */
    apex_event_type trigger;                          /*!< the event that triggers the tuning update */
    apex_tuning_session_handle tuning_session_handle; /*!< the Active Harmony tuning session handle */
    bool running;                                     /*!< the current state of the tuning */
    apex_ah_tuning_strategy strategy;                 /*!< the requested Active Harmony tuning strategy */
} apex_tuning_request_struct;
```

## Environment variables

Please see the [environment variables](environment.md) section of the
documentation. Please note that all environment variables can also be
queried or set at runtime with associated API calls. For example, the 
APEX_CSV_OUTPUT variable can also be set/queried with:

``` c
void apex_set_csv_output (int);
int apex_get_csv_output (void);
```

## General Utility functions

### Initialization

``` c++
/* C++ */
void apex::init (const char *thread_name);
```
``` c
/* C */
void apex_init (const char *thread_name);
```

APEX initialization is required to set up data structures and spawn the
necessary helper threads, including the background system state query thread,
the policy engine thread, and the profile handler thread. The thread name
parameter will be used as the top-level timer for the the main thread of
execution.

### Finalization

``` c++
/* C++ */
void apex::finalize (void);
```
``` c
/* C */
void apex_finalize (void);
```

APEX finalization is required to format any desired output (screen, csv,
profile, etc.) and terminate all APEX helper threads. No memory is freed at
this point - that is done by the `apex_cleanup()` call.  The reason for this is
that applications may want to perform reporting after finalization, so the
performance state of the application should still exist. 

### Cleanup

``` c++
/* C++ */
void apex::cleanup (void);
```
``` c
/* C */
void apex_cleanup (void);
```

APEX cleanup frees all memory associated with APEX. 

### Setting node ID

``` c++
/* C++ */
void apex::set_node_id (const uint64_t id);
```
``` c
/* C */
void apex_set_node_id (const uint64_t id);
```

When running in distributed environments, assign the specified id number as the
APEX node ID. This can be an MPI rank or an HPX locality, for example. 

### Registering threads

``` c++
/* C++ */
void apex::register_thread (const std::string &name);
```
``` c
/* C */
void apex_register_thread (const char *name);
```

Register a new OS thread with APEX.  This method should be called whenever a
new OS thread is spawned by the application or the runtime.  An empty string
or null string is valid input.

### Exiting a thread

``` c++
/* C++ */
void apex::exit_thread (void);
```
``` c
/* C */
void apex_exit_thread (void);
```

Before any thread other than the main thread of execution exits, notify APEX
that the thread is exiting. The main thread should not call this function, but
apex_finalize instead. Exiting the thread will trigger an event in APEX, so 
any policies associated with a thread exit will be executed.

### Getting the APEX version

``` c++
/* C++ */
std::string & apex::version (void);
```
``` c
/* C */
const char * apex_version (void);
```

Return the APEX version as a string.

### Getting the APEX settings

``` c++
/* C++ */
std::string & apex::get_options (void);
```
``` c
/* C */
const char * apex_get_options (void);
```

Return the current APEX options as a string.

## Basic measurement Functions (introspection)

### Starting a timer

``` c++
/* C++ */
apex_profiler_handle apex::start (const std::string &timer_name);
apex_profiler_handle apex::start (const apex_function_address function_address);
```
``` c
/* C */
apex_profiler_handle apex_start (apex_profiler_type type, const void * identifier);
```

Create an APEX timer and start it. An APEX profiler object is returned,
containing an identifier that APEX uses to stop the timer.  The timer is either
identified by a name or a function/task instruction pointer address.

### Stopping a timer

``` c++
/* C++ */
void apex::stop (apex_profiler_handle the_profiler);
```
``` c
/* C */
void apex_stop (apex_profiler_handle the_profiler);
```

The timer associated with the profiler object is stopped and placed on an
internal queue to be processed by the profiler handler thread in the
background.  The profiler object is flagged as "stopped", so that when the
profiler is processed the call count for this particular timer will be
incremented by 1, *unless* the timer was started by `apex_resume()` (see
below). The profiler handle will be freed internally by APEX after processing.

### Yielding a timer

``` c++
/* C++ */
void apex::yield (apex_profiler_handle the_profiler);
```
``` c
/* C */
void apex_yield (apex_profiler_handle the_profiler);
```

The timer associated with the profiler object is stopped and placed on an
internal queue to be processed by the profiler handler thread in the
background.  The profiler object is flagged as *NOT stopped*, so that when the
profiler is processed the call count will NOT be incremented.  An application
using apex_yield should not use apex_resume to restart the timer, it should use
apex_start. `apex_yield()` is intended for situations when the completion state
of the task is known and the state is *not complete*.
below). The profiler handle will be freed internally by APEX after processing.

### Resuming a timer

``` c++
/* C++ */
apex_profiler_handle apex::resume (const std::string &timer_name);
apex_profiler_handle apex::resume (const apex_function_address function_address);
```
``` c
/* C */
apex_profiler_handle apex_resume (apex_profiler_type type, const void * identifier);
```

Create an APEX timer and start it. An APEX profiler object is returned,
containing an identifier that APEX uses to stop the timer.  The profiler is
flagged as *NOT a new task*, so that when it is stopped by apex_stop the call
count for this particular timer will not be incremented. Apex_resume is intended
for situations when the completion state of a task is NOT known when control
is returned to the task scheduler, but is known when an interrupted task is 
resumed.

### Creating a new task dependency

``` c++
/* C++ */
void apex::new_task (std::string & name, const void * task_id);
void apex::new_task (const apex_function_address function_address, const void * task_id);
```
``` c
/* C */
void apex_new_task (apex_profiler_type type, const void * identifier, const void * task_id)
```

Register the creation of a new task. This is used to track task dependencies in
APEX. APEX assumes that the current APEX profiler refers to the task that is
the parent of this new task. The task_info object is a generic pointer to
whatever data might need to be passed to a policy executed on when a new task
is created.

### Sampling a value

``` c++
/* C++ */
void apex::sample_value (const std::string & name, const double value)
```
``` c
/* C */
void apex_sample_value (const char * name, const double value);
```

Record a measurement of the specified counter with the specified value. For
example, "bytes transferred" and "1024".

### Setting the OS thread state

``` c++
/* C++ */
void apex::set_state (apex_thread_state state);
```
``` c
/* C */
void apex_set_state (apex_thread_state state);
```

Set the state of the current OS thread.  States can include things like idle,
busy, waiting, throttled, blocked.

## Policy-related methods (adaptation)

### Registering an event-based policy function

``` c++
/* C++ */
apex_policy_handle apex::register_policy (const apex_event_type when, std::function<int(apex_context const&)> f);
std::set<apex_policy_handle> apex::register_policy (std::set<apex_event_type> when, std::function<int(apex_context const&)> f);
```
``` c
/* C */
apex_policy_handle apex_register_policy (const apex_event_type when, int(*f)(apex_context const&));
```

APEX provides the ability to call an application-specified function when
certain events occur in the APEX library, or periodically. This assigns the
passed in function to the event, so that when that event occurs in APEX, the
function is called. The context for the event will be passed to the registered
function. A set of events can also be used to register a policy function, which will
return a set of policy handles. When any event in the set occurs, the function will
be called.

### Registering a periodic policy

``` c++
/* C++ */
apex_policy_handle apex::register_periodic_policy(const unsigned long period, std::function<int(apex_context const&)> f);
```
``` c
/* C */
apex_policy_handle apex_register_periodic_policy (const unsigned long period, int(*f)(apex_context const&));
```

Apex provides the ability to call an application-specified function
periodically. This method assigns the passed in function to be called on a
periodic basis. The context for the event will be passed to the registered
function.  The period units are in microseconds (us).

### De-registering a policy

``` c++
/* C++ */
apex::deregister_policy (apex_policy_handle handle);
```
``` c
/* C */
apex_deregister_policy (apex_policy_handle handle);
```

Remove the specified policy so that it will no longer be executed, whether it
is event-based or periodic. The calling code should not try to dereference the
policy handle after this call, as the memory pointed to by the handle will be
freed.

### Registering a custom event

``` c++
/* C++ */
apex_event_type apex::register_custom_event (const std::string & name);
```
``` c 
/* C */
apex_event_type apex_register_custom_event (const char * name);
```

Register a new event type with APEX.

### Trigger a custom event

``` c++
/* C++ */
void apex::custom_event (apex_event_type event_type, const void * event_data);
```
``` c
/* C */
void apex_custom_event (const char * name, const void * event_data);
```

Trigger a custom event.  This function will pass a custom event to the APEX
event listeners. Each listeners' custom event handler will handle the custom
event. Policy functions will be passed the custom event name in the event
context.  The event data pointer is to be used to pass memory to the policy
function from the code that triggered the event.

### Request a profile from APEX 

``` c++
/* C++ */
apex_profile * apex::get_profile (const std::string & name);
apex_profile * apex::get_profile (const apex_function_address function_address);
```
``` c
/* C */
apex_profile * apex_get_profile (apex_profiler_type type, const void * identifier)
```

This function will return the current profile for the specified identifier.
Because profiles are updated out-of-band, it is possible that this profile
values are out of date. This profile can be either a timer or a sampled value.

### Reset a profile

``` c++
/* C++ */
void apex::reset (const std::string & timer_name);
void apex::reset (const apex_function_address function_address);
```
``` c
/* C */
void apex_reset (apex_profiler_type type, const void * identifier)
```

This function will reset the profile associated with the specified timer or
counter id to zero.  If the identifier is null, all timers and counters will be
reset.

## Concurrency Throttling Policy Functions

### Setup tuning for adaptation

``` c++
/* C++ */
apex_tuning_session_handle setup_custom_tuning(apex_tuning_request & request);
```
``` c
apex_tuning_session_handle setup_custom_tuning(apex_tuning_request * request);
```

Setup tuning of specified parameters to optimize for a custom metric, using
multiple input criteria.  This function will initialize a policy to optimize a
custom metric, using the list of tunable parameters. The system tries to
minimize the custom metric. After evaluating the state of the system, the
policy will assign new values to the inputs.

### Get the current thread cap

``` c++
/* C++ */
int apex::get_thread_cap (void);
```
``` c
/* C */
int apex_get_thread_cap (void);
```

This function will return the current thread cap based on the throttling policy.

### Set the current thread cap

``` c++
/* C++ */
void apex::set_thread_cap (int new_cap);
```
``` c
/* C */
void apex_set_thread_cap (int new_cap);
```

This function will set the current thread cap based on an external throttling policy.

## Event-based API (OCR, Legion support - *TBD*)

The OCR and Legion runtimes teams have met to propose a common API for
measuring asynchronous task-based runtimes.
For more details, see <https://github.com/khuck/xpress-apex/issues/37>.

``` c++
/* C++ */
apex::task_create (uint64_t parent_id)
apex::dependency_reached (uint64_t event_id, uint64_t data_id, uint64_t task_id, uint64_t parent_id, ?)
apex::task_ready (uint64_t why_ready)
apex::task_execute (uint64_t why_delay, const apex_function_address function)
apex::task_finished (uint64_t task_id)
apex::task_destroy (uint64_t task_id)
apex::data_create (uint64_t data_id)
apex::data_new_size (uint64_t data_id)
apex::data_move_from (uint64_t data_id, uint64_t target_location)
apex::data_move_to (uint64_t data_id, uint64_t source_location)
apex::data_replace (uint64_t data_id, uint64_t new_id)
apex::data_destroy (uint64_t data_id)
apex::event_create (uint64_t event_id, parent_task_id)
apex::event_add_dependency (uint64_t event_id, uint64_t data_event_task_id, uint64_t parent_task_id)
apex::event_trigger (uint64_t event_id)
apex::event_destroy (uint64_t event_id)
```
``` c
/* C API tbd */
```
