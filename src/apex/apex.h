/*  Copyright (c) 2014 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/*
 * APEX external API
 *
 */

/*
 * The C API is not required for HPX support. 
 * But don't delete it just yet. 
 */

#ifndef APEX_H
#define APEX_H

#include "apex_types.h"
#include "apex_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialization, finalization functions
 */
APEX_EXPORT void apex_init(const char * thread_name);
APEX_EXPORT void apex_init_args(int argc, char** argv, const char * thread_name);
APEX_EXPORT void apex_finalize();

/*
 * Functions for starting, stopping timers
 */
APEX_EXPORT apex_profiler_handle apex_start_name(const char * timer_name);
APEX_EXPORT apex_profiler_handle apex_start_address(apex_function_address function_address);
APEX_EXPORT void apex_stop_profiler(apex_profiler_handle profiler);
APEX_EXPORT void apex_resume(apex_profiler_handle profiler);

/*
 * Functions for resetting timer values
 */
APEX_EXPORT void apex_reset_name(const char * timer_name);
APEX_EXPORT void apex_reset_address(apex_function_address function_address);

/*
 * Function for sampling a counter value
 */
APEX_EXPORT void apex_sample_value(const char * name, double value);

/*
 * Utility functions
 */
APEX_EXPORT void apex_set_node_id(int id);
APEX_EXPORT double apex_version(void);
APEX_EXPORT void apex_node_id(int id);
APEX_EXPORT void apex_register_thread(const char * name);

/*
 * Power-related functions
 */
APEX_EXPORT void apex_track_power(void);
APEX_EXPORT void apex_track_power_here(void);
APEX_EXPORT void apex_enable_tracking_power(void);
APEX_EXPORT void apex_disable_tracking_power(void);
APEX_EXPORT void apex_set_interrupt_interval(int seconds);

/*
 * Policy Engine functions.
 */
APEX_EXPORT apex_policy_handle apex_register_policy(const apex_event_type when, int (*f)(apex_context const));
APEX_EXPORT apex_policy_handle apex_register_periodic_policy(unsigned long period, int (*f)(apex_context const));

/*
 */
APEX_EXPORT apex_profile * apex_get_profile_from_name(const char * timer_name);
APEX_EXPORT apex_profile * apex_get_profile_from_address(apex_function_address function_address);

#define apex_macro(name, member_variable, type, default_value) \
void apex_set_##member_variable (type inval); \
type apex_get_##member_variable (void);
FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#ifdef __cplusplus
}
#endif

#endif //APEX_H
