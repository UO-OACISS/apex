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

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialization, finalization functions
 */
void apex_init(const char * thread_name);
void apex_init_args(int argc, char** argv, const char * thread_name);
void apex_finalize();

/*
 * Functions for starting, stopping timers
 */
apex_profiler_handle apex_start_name(const char * timer_name);
apex_profiler_handle apex_start_address(apex_function_address function_address);
void apex_stop(apex_profiler_handle profiler);
void apex_resume(apex_profiler_handle profiler);

/*
 * Function for sampling a counter value
 */
void apex_sample_value(const char * name, double value);

/*
 * Utility functions
 */
void apex_set_node_id(int id);
double apex_version(void);
void apex_node_id(int id);
void apex_register_thread(const char * name);

/*
 * Power-related functions
 */
void apex_track_power(void);
void apex_track_power_here(void);
void apex_enable_tracking_power(void);
void apex_disable_tracking_power(void);
void apex_set_interrupt_interval(int seconds);

/*
 * Policy Engine functions.
 */
apex_policy_handle apex_register_policy(const apex_event_type when, int (*f)(apex_context const));
apex_policy_handle apex_register_periodic_policy(unsigned long period, int (*f)(apex_context const));

/*
 */
apex_profile * apex_get_profile_from_name(const char * timer_name);
apex_profile * apex_get_profile_from_address(apex_function_address function_address);

#define apex_macro(name, member_variable, type, default_value) \
void apex_set_##member_variable (type inval); \
type apex_get_##member_variable (void);
FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#ifdef __cplusplus
}
#endif

#endif //APEX_H
