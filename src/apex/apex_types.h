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
typedef enum _event_type {
  STARTUP,
  SHUTDOWN,
  NEW_NODE,
  NEW_THREAD,
  START_EVENT,
  RESUME_EVENT,
  STOP_EVENT,
  SAMPLE_VALUE,
  PERIODIC
} apex_event_type;

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
  TIMER,
  COUNTER
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

/** Rather than use void pointers everywhere, be explicit about
 * what the functions are expecting.
 */
typedef void * apex_function_address; // generic function pointer

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define FOREACH_APEX_OPTION(macro) \
    macro (APEX_TAU, use_tau, bool, false) \
    macro (APEX_POLICY, use_policy, bool, true) \
    macro (APEX_CONCURRENCY, use_concurrency, int, 0) \
    macro (APEX_CONCURRENCY_PERIOD, concurrency_period, int, 1000000) \
    macro (APEX_SCREEN_OUTPUT, use_screen_output, bool, true) \
    macro (APEX_PROFILE_OUTPUT, use_profile_output, bool, false) \

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
