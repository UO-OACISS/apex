//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_TYPES_HPP
#define APEX_TYPES_HPP

/* required for Doxygen */

/** @file */ 

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

/** The address of a C++ object in APEX.
 * Not useful for the caller that gets it back, but required
 * for stopping the timer later.
 */
typedef void* apex_profiler_handle; // address of internal C++ object

/** A null pointer representing an APEX profiler handle.
 * Used when a null APEX profile handle is to be passed in to
 * apex::stop when the profiler object wasn't retained locally.
 */
#define APEX_NULL_PROFILER_HANDLE (apex_profiler_handle)(NULL) // for comparisons

/** Rather than use void pointers everywhere, be explicit about
 * what the functions are expecting.
 */
//typedef int (*apex_function_address)(void); // generic function pointer
typedef uintptr_t apex_function_address; // generic function pointer

/**
 * Typedef for enumerating the different timer types
 */
typedef enum _apex_profiler_type {
    APEX_FUNCTION_ADDRESS = 0, /*!< The ID is a function (or instruction) address */
    APEX_NAME_STRING           /*!< The ID is a character string */
} apex_profiler_type;

/**
 * Typedef for enumerating the different event types
 */
typedef enum _error_codes {
  APEX_NOERROR = 0, /*!< No error occurred */
  APEX_ERROR        /*!< Some error occurred - check stderr output for details */
} apex_error_code;

#define APEX_MAX_EVENTS 128 /*!< The maximum number of event types. Allows for ~20 custom events. */

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
 * Typedef for enumerating the thread states. 
 */
typedef enum _thread_state {
    APEX_IDLE,          /*!< Thread is idle */
    APEX_BUSY,          /*!< Thread is working */
    APEX_THROTTLED,     /*!< Thread is throttled (sleeping) */
    APEX_WAITING,       /*!< Thread is waiting for a resource */
    APEX_BLOCKED        /*!< Thread is blocked */
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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 * Structure that holds a profiler ID
 */
typedef struct _apex_profiler_id
{
    apex_profiler_type type;
    union {
      apex_function_address address;
      const char * name;
    } identifier;
} apex_profiler_id;

#endif //DOXYGEN_SHOULD_SKIP_THIS

/** A reference to the policy object,
 * so that policies can be "unregistered", or paused later
 */
typedef struct _policy_handle
{
    int id;           /*!< The ID of the policy, used internally to APEX */
    apex_event_type event_type;    /*!< The type of policy */
    unsigned long period;       /*!< If periodic, the length of the period */
} apex_policy_handle;

/** The APEX context when an event occurs.
 * 
 */
typedef struct _context
{
    apex_event_type event_type;        /*!< The type of the event currently
                                           processing */
    apex_policy_handle* policy_handle; /*!< The policy handle for the current
                                           policy function */
    void * data;  /*!< Data associated with the event, such as the custom_data
                       for a custom_event */
} apex_context;

/** The type of a profiler object
 * 
 */
typedef enum _profile_type {
  APEX_TIMER,        /*!< This profile is a instrumented timer */
  APEX_COUNTER       /*!< This profile is a sampled counter */
} apex_profile_type;

/**
 * The profile object for a timer in APEX.
 */
typedef struct _profile
{
    double calls;         /*!< Number of times a timer was called, or the number
                              of samples collected for a counter */
    double accumulated;   /*!< Accumulated values for all calls/samples */
    double sum_squares;   /*!< Running sum of squares calculation for all
                              calls/samples */
    double minimum;       /*!< Minimum value seen by the timer or counter */
    double maximum;       /*!< Maximum value seen by the timer or counter */
    apex_profile_type type; /*!< Whether this is a timer or a counter */
} apex_profile;

/** Rather than use void pointers everywhere, be explicit about
 * what the functions are expecting.
 */
typedef int (*apex_policy_function)(apex_context const context);

/**
 *  A handle to a tuning session.
 */
typedef uint32_t apex_tuning_session_handle;

/** A null pointer representing an APEX function address.
 * Used when a null APEX function address is to be passed in to
 * any apex functions to represent "all functions". 
 */
#define APEX_NULL_FUNCTION_ADDRESS 0L // for comparisons
//#define APEX_NULL_FUNCTION_ADDRESS (apex_function_address)(NULL) // for comparisons

/**
 * Special profile counter for derived idle time 
 **/
#define APEX_IDLE_TIME "APEX Idle"
/**
 * Special profile counter for derived non-idle time 
 **/
#define APEX_NON_IDLE_TIME "APEX Non-Idle"
/**
 * Special profile counter for derived idle rate 
 **/
#define APEX_IDLE_RATE "APEX Idle Rate"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifdef APEX_HAVE_TAU
    #ifdef APEX_HAVE_HPX3
        // don't enable TAU by default for HPX.
        #define APEX_TAU_DEFAULT false
    #else
        #define APEX_TAU_DEFAULT true
    #endif
#else
    #define APEX_TAU_DEFAULT false
#endif

#define FOREACH_APEX_OPTION(macro) \
    macro (APEX_TAU, use_tau, bool, APEX_TAU_DEFAULT) \
    macro (APEX_POLICY, use_policy, bool, true) \
    macro (APEX_MEASURE_CONCURRENCY, use_concurrency, int, 0) \
    macro (APEX_UDP_SINK, use_udp_sink, bool, false) \
    macro (APEX_MEASURE_CONCURRENCY_PERIOD, concurrency_period, int, 1000000) \
    macro (APEX_SCREEN_OUTPUT, use_screen_output, bool, false) \
    macro (APEX_PROFILE_OUTPUT, use_profile_output, int, false) \
    macro (APEX_TASKGRAPH_OUTPUT, use_taskgraph_output, bool, false) \
    macro (APEX_PROC_CPUINFO, use_proc_cpuinfo, bool, false) \
    macro (APEX_PROC_MEMINFO, use_proc_meminfo, bool, false) \
    macro (APEX_PROC_NET_DEV, use_proc_net_dev, bool, false) \
    macro (APEX_PROC_SELF_STATUS, use_proc_self_status, bool, false) \
    macro (APEX_PROC_STAT, use_proc_stat, bool, true) \
    macro (APEX_THROTTLE_CONCURRENCY, throttle_concurrency, bool, false) \
    macro (APEX_THROTTLING_MAX_THREADS, throttling_max_threads, int, sysconf( _SC_NPROCESSORS_ONLN )) \
    macro (APEX_THROTTLING_MIN_THREADS, throttling_min_threads, int, 1) \
    macro (APEX_THROTTLE_ENERGY, throttle_energy, bool, false) \
    macro (APEX_THROTTLING_MAX_WATTS, throttling_max_watts, int, 300) \
    macro (APEX_THROTTLING_MIN_WATTS, throttling_min_watts, int, 150) \
    macro (APEX_PTHREAD_WRAPPER_STACK_SIZE, pthread_wrapper_stack_size, int, 0)

#define FOREACH_APEX_STRING_OPTION(macro) \
    macro (APEX_UDP_SINK_HOST, udp_sink_host, char*, "localhost") \
    macro (APEX_UDP_SINK_PORT, udp_sink_port, char*, "5560") \
    macro (APEX_UDP_SINK_CLIENTIP, udp_sink_clientip, char*, "127.0.0.1") \
    macro (APEX_PAPI_METRICS, papi_metrics, char*, "") \
    macro (APEX_PLUGINS, plugins, char*, "") \
    macro (APEX_PLUGINS_PATH, plugins_path, char*, "./") \

#if defined(__linux)
#  define APEX_NATIVE_TLS __thread
#elif defined(_WIN32) || defined(_WIN64)
#  define APEX_NATIVE_TLS __declspec(thread)
#elif defined(__FreeBSD__) || (defined(__APPLE__) && defined(__MACH__))
#  define APEX_NATIVE_TLS __thread
#else
#  error "Native thread local storage is not supported for this platform"
#endif

// This macro is to prevent compiler warnings for stub implementations,
// in particular empty virtual implementations.
#define APEX_UNUSED(expr) do { (void)(expr); } while (0)

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif //APEX_TYPES_HPP
