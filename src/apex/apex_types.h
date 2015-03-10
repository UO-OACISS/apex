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

/**
 * Typedef for enumerating the different event types
 */
typedef enum _error_codes {
  APEX_NOERROR = 0,
  APEX_ERROR
} apex_error_code;

/**
 * Typedef for enumerating the different event types
 */
typedef enum _event_type {
  APEX_STARTUP,
  APEX_SHUTDOWN,
  APEX_NEW_NODE,
  APEX_NEW_THREAD,
  APEX_START_EVENT,
  APEX_RESUME_EVENT,
  APEX_STOP_EVENT,
  APEX_SAMPLE_VALUE,
  APEX_PERIODIC,
  APEX_CUSTOM_EVENT
} apex_event_type;

/**
 * Typedef for enumerating the different optimization strategies
 * for throttling.
 */
typedef enum {APEX_MAXIMIZE_THROUGHPUT,
	          APEX_MAXIMIZE_ACCUMULATED, 
	          APEX_MINIMIZE_ACCUMULATED
} apex_optimization_criteria_t;

/**
 * Typedef for enumerating the different optimization methods
 * for throttling.
 */
typedef enum {APEX_SIMPLE_HYSTERESIS,
	          APEX_DISCRETE_HILL_CLIMBING, 
	          APEX_ACTIVE_HARMONY
} apex_optimization_method_t;

/** A reference to the policy object,
 * so that policies can be "unregistered", or paused later
 */
typedef struct _policy_handle
{
    int id;
} apex_policy_handle;

/** The APEX context when an event occurs.
 * 
 */
typedef struct _context
{
    apex_event_type event_type;
    apex_policy_handle* policy_handle;
} apex_context;

/** The type of a profiler object
 * 
 */
typedef enum _profile_type {
  APEX_TIMER,
  APEX_COUNTER
} apex_profile_type;

/**
 * The profile object for a timer in APEX.
 */
typedef struct _profile
{
    double calls;
    double accumulated;
    double sum_squares;
    double minimum;
    double maximum;
    apex_profile_type type;
} apex_profile;

/** The address of a C++ object in APEX.
 * Not useful for the caller that gets it back, but required
 * for stopping the timer later.
 */
typedef uintptr_t apex_profiler_handle; // address of internal C++ object

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

/** Rather than use void pointers everywhere, be explicit about
 * what the functions are expecting.
 */
typedef int (*apex_policy_function)(apex_context const context);

/** A null pointer representing an APEX function address.
 * Used when a null APEX function address is to be passed in to
 * any apex functions to represent "all functions". 
 */
#define APEX_NULL_FUNCTION_ADDRESS 0L // for comparisons
//#define APEX_NULL_FUNCTION_ADDRESS (apex_function_address)(NULL) // for comparisons

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define FOREACH_APEX_OPTION(macro) \
    macro (APEX_TAU, use_tau, bool, false) \
    macro (APEX_POLICY, use_policy, bool, true) \
    macro (APEX_CONCURRENCY, use_concurrency, int, 0) \
    macro (APEX_CONCURRENCY_PERIOD, concurrency_period, int, 1000000) \
    macro (APEX_SCREEN_OUTPUT, use_screen_output, bool, true) \
    macro (APEX_PROFILE_OUTPUT, use_profile_output, int, 0) \

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
