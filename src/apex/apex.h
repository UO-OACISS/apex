/*  Copyright (c) 2014 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* required for Doxygen */

/** @file */ 

/*
 * APEX external API
 *
 */

/*
 * The C API is required for HPX5 support. 
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

/* The functions from here on should all be documented by Doxygen. */

/**
 \brief Intialize APEX.
 \warning For best results, this function should be called before any other 
          APEX functions. 
 \warning Use this version of apex_init when you do not have access
          to the input arguments.
 
 \param thread_name The name of the thread, or NULL. The lifetime of the
                    thread will be timed with a timer using this same name.
 \return No return value.
 */
APEX_EXPORT void apex_init(const char * thread_name);

/**
 \brief Intialize APEX.
 \warning For best results, this function should be called before any other 
          APEX functions. 
 \warning Use this version of apex_init when you have access
          to the input arguments.
 
 \param argc The number of arguments passed in to the program.
 \param argv An array of arguments passed in to the program.
 \param thread_name The name of the thread, or NULL. The lifetime of the
                    thread will be timed with a timer using this same name.
 \return No return value.
 */
APEX_EXPORT void apex_init_args(int argc, char** argv, const char * thread_name);

/**
 \brief Finalize APEX.
 \warning For best results, this function should be explicitly called 
          before program exit. If not explicitly called from the 
		  application or runtime, it will be automatically
		  called when the APEX main singleton object is destructed,
		  but there are no guarantees that will work correctly.
 
 The finalization method will terminate all measurement and optionally:
 - print a report to the screen
 - write a TAU profile to disk
 \return No return value.
 */
APEX_EXPORT void apex_finalize();

/*
 * Functions for starting, stopping timers
 */

/**
 \brief Start a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the name
 passed in to this function.
 
 \param timer_name The name of the timer.
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
		 possible, and passed in to the matching apex_stop_name()
		 call when the timer should be stopped.
 \sa apex_stop_name
 */
APEX_EXPORT apex_profiler_handle apex_start_name(const char * timer_name);

/**
 \brief Start a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the 
 address passed in to this function.
 
 \param function_address The address of the function to be timed
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
		 possible, and passed in to the matching apex_stop_profiler()
		 call when the timer should be stopped.
 \sa apex_stop_profiler
 */
APEX_EXPORT apex_profiler_handle apex_start_address(apex_function_address function_address);

/**
 \brief Stop a timer.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process.
 
 \param profiler The handle of the profiler object.
 \return No return value.
 \sa apex_start_name, apex_start_address
 */
APEX_EXPORT void apex_stop_profiler(apex_profiler_handle profiler);

/**
 \brief Stop a timer, but don't increment the number of calls.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process. The number of calls
 will NOT be incremented - this "task" was yielded, not completed.
 It will be resumed by another thread at a later time.
 
 \param profiler The handle of the profiler object.
 \return No return value.
 \sa apex_start_name, apex_start_address
 */
APEX_EXPORT void apex_yield_profiler(apex_profiler_handle profiler);

/**
 \brief Resume a timer.

 This function will restart the specified profiler object. The
 difference between this function and the apex_start_name or
 apex_start_address functions is that the number of calls to that
 timer will not be incremented.
 
 \param profiler The handle of the profiler object.
 \return No return value.
 \sa apex_start_name, apex_start_address, apex_stop_profiler
 */
APEX_EXPORT void apex_resume(apex_profiler_handle profiler);

/*
 * Functions for resetting timer values
 */

/**
 \brief Reset a timer.

 This function will reset the profile associated with the specified
 timer to zero.
 
 \param timer_name The name of the timer.
 \return No return value.
 */
APEX_EXPORT void apex_reset_name(const char * timer_name);

/**
 \brief Reset a timer.

 This function will reset the profile associated with the specified
 timer to zero.
 
 \param function_address The function address of the timer.
 \return No return value.
 */
APEX_EXPORT void apex_reset_address(apex_function_address function_address);

/**
 \brief Set the thread state

 This function will set the thread state in APEX for 3rd party observation
 
 \param state The state of the thread.
 \return No return value.
 */
APEX_EXPORT void apex_set_state(apex_thread_state state);

/*
 * Function for sampling a counter value
 */

/**
 \brief Sample a state value.

 This function will retain a sample of some value. The profile
 for this sampled value will store the min, mean, max, total
 and standard deviation for this value for all times it is sampled.
 
 \param name The name of the sampled value
 \param value The sampled value
 \return No return value.
 */
APEX_EXPORT void apex_sample_value(const char * name, double value);

/**
 \brief Trigger a custom event.

 This function will pass a custom event to the APEX event listeners.
 Each listeners' on_custom() event will handle the custom event.
 Policy functions will be passed the custom event name in the event context.
 
 \param event_name The name of the custom event
 \param custom_data Data specific to the custom event
 \return No return value.
 */
APEX_EXPORT void apex_custom_event(const char * event_name, void * custom_data);

/*
 * Utility functions
 */


/**
 \brief Set this process' node ID.

 For distributed applications, this function will store the
 node ID. Common values are the MPI rank, the HPX locality, etc.
 This ID will be used to identify the process in the global
 performance space.
 
 \param id The node ID for this process.
 \return No return value.
 */
APEX_EXPORT void apex_set_node_id(int id);

/**
 \brief Return the APEX version.
 
 \return A character string with the APEX version. This string
 should be freed after the calling function is done with it.
 */
APEX_EXPORT const char * apex_version(void);

/**
 \brief Set this process' node ID.

 For distributed applications, this function will store the
 node ID. Common values are the MPI rank, the HPX locality, etc.
 This ID will be used to identify the process in the global
 performance space.
 
 \param id The node ID for this process.
 \return No return value.
 */
APEX_EXPORT void apex_node_id(int id);

/**
 \brief Register a new thread.

 For multithreaded applications, register a new thread with APEX.
 \warning Failure to register a thread with APEX may invalidate
 statistics, and may prevent the ability to use timers or sampled
 values for this thread.
 
 \param name The name that will be assigned to the new thread.
 \return No return value.
 */
APEX_EXPORT void apex_register_thread(const char * name);

#ifndef DOXYGEN_SHOULD_SKIP_THIS // not sure if these will stay in the API

/*
 * Power-related functions
 */
APEX_EXPORT void apex_track_power(void);
APEX_EXPORT void apex_track_power_here(void);
APEX_EXPORT void apex_enable_tracking_power(void);
APEX_EXPORT void apex_disable_tracking_power(void);
APEX_EXPORT void apex_set_interrupt_interval(int seconds);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/*
 * Policy Engine functions.
 */

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 when certain events occur in the APEX library, or periodically.
 This assigns the passed in function to the event, so that when that
 event occurs in APEX, the function is called. The context for the
 event will be passed to the registered function.
 
 \param when The APEX event when this function should be called
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 */
APEX_EXPORT apex_policy_handle apex_register_policy(const apex_event_type when, apex_policy_function f);

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 periodically.  This assigns the passed in function to be called on a periodic
 basis.  The context for the event will be passed to the registered function.
 
 \param period How frequently the function should be called
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 */
APEX_EXPORT apex_policy_handle apex_register_periodic_policy(unsigned long period, apex_policy_function f);

/**
 \brief Get the current profile for the specified function address.

 This function will return the current profile for the specified address.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date.  This profile can be either a timer or a sampled value.
 
 \param timer_name The name of the function
 \return The current profile for that timed function or sampled value.
 */
APEX_EXPORT apex_profile * apex_get_profile_from_name(const char * timer_name);

/**
 \brief Get the current profile for the specified function address.

 This function will return the current profile for the specified address.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date. 
 
 \param function_address The address of the function.
 \return The current profile for that timed function.
 */
APEX_EXPORT apex_profile * apex_get_profile_from_address(apex_function_address function_address);

/**
 \brief Get the current power reading

 This function will return the current power level for the node, measured in Watts.

 \return The current power level in Watts.
 */
APEX_EXPORT double apex_current_power_high(void);

/**
 \brief Initialize the power cap throttling policy.

 This function will initialize APEX for power cap throttling. There are several
 environment variables that control power cap throttling:

 <dl>
 <dt> HPX_THROTTLING </dt>
 <dd> If set, throttling will be enabled and initialized at startup.</dd>
 <dt> APEX_THROTTLING_MAX_THREADS </dt>
 <dd> The maximum number of threads the throttling system will allow. The default
      value is 48. </dd>
 <dt> APEX_THROTTLING_MIN_THREADS </dt>
 <dd> The minimum number of threads the throttling system will allow. The default
      value is 12.  </dd>
 <dt> APEX_THROTTLING_MAX_WATTS </dt>
 <dd> The maximum number of Watts the system can consume as an average rate. The
      default value is 220. </dd>
 <dt> APEX_THROTTLING_MIN_WATTS </dt>
 <dd> The minimum number of Watts the system can consume as an average rate. The
      default value is 180. </dd>
 <dt> HPX_ENERGY_THROTTLING </dt>
 <dd> If set, power/energy throttling will be performed.  </dd>
 <dt> HPX_ENERGY </dt>
 <dd> TBD </dd>
 </dl>

 After evaluating the state of the system, the policy will set the thread cap,
 which can be queried using @ref apex_get_thread_cap().
 
 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int apex_setup_power_cap_throttling(void);      // initialize

/**
 \brief Setup throttling to optimize for the specified function.

 This function will initialize the throttling policy to optimize for the 
 specified function. The optimization criteria include maximizing throughput,
 minimizing or maximizing time spent in the specified function. After
 evaluating the state of the system, the policy will set the thread cap, which
 can be queried using @ref apex_get_thread_cap().

 \param the_address The address of the function to be optimized.
 \param criteria The optimization criteria.
 \param method The optimization method.
 \param update_interval The time between observations, in microseconds.
 \return APEX_NOERROR on success, otherwise an error code.
 */

APEX_EXPORT int apex_setup_timer_address_throttling(apex_function_address the_address,
        apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval);

/**
 \brief Setup throttling to optimize for the specified function or counter.

 This function will initialize the throttling policy to optimize for the 
 specified function or counter. The optimization criteria include maximizing
 throughput, minimizing or maximizing time spent in the specified function
 or value sampled in the counter. After evaluating the state of the system,
 the policy will set the thread cap, which can be queried using 
 @ref apex_get_thread_cap().

 \param the_name The name of the function or counter to be optimized.
 \param criteria The optimization criteria.
 \param method The optimization method.
 \param update_interval The time between observations, in microseconds.

 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int apex_setup_timer_name_throttling(const char * the_name,
        apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval);

/**
 \brief Terminate the throttling policy.

 This function will terminate the throttling policy.

 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int apex_shutdown_throttling(void);   // terminate

/**
 \brief Get the current thread cap set by the throttling.

 This function will return the current thread cap based on the throttling
 policy.

 \return The current thread cap value.
 */
APEX_EXPORT int apex_get_thread_cap(void);             // for thread throttling


#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define apex_macro(name, member_variable, type, default_value) \
void apex_set_##member_variable (type inval); \
type apex_get_##member_variable (void);
FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
}
#endif

#endif //APEX_H

