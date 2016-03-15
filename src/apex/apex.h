/*  Copyright (c) 2014 University of Oregon
 *
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
 \sa @ref apex_init_args, @ref apex_finalize
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
 \sa @ref apex_init, @ref apex_finalize
 */
APEX_EXPORT void apex_init_args(int argc, char** argv, const char * thread_name);

/**
 \brief Finalize APEX.
 
 The stop measurement method will terminate all measurement and optionally:
 - print a report to the screen
 - write a TAU profile to disk
 \return No return value.
 \sa @ref apex_init, @ref apex_init_args
 */
APEX_EXPORT void apex_finalize();

/**
 \brief Cleanup APEX.
 \warning For best results, this function should be explicitly called 
                    to free all memory allocated by APEX. If not explicitly called from
                    the application or runtime, it will be automatically called when the
                    APEX main singleton object is destructed. apex_finalize will be 
                    automatically called from apex_cleanup if it has not yet been called.
 
 The cleanup method will free all allocated memory for APEX.
 \return No return value.
 \sa @ref apex_finalize
 */
APEX_EXPORT void apex_cleanup();

/*
 * Functions for starting, stopping timers
 */

/**
 \brief Start a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the address
 or name passed in to this function.  If both are zero (null) then the call
 will fail and the return value will be null.
 
 \param type The type of the address to be stored. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be timed, or a "const
             char *" pointer to the name of the timer.
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching @ref apex_stop
         call when the timer should be stopped.
 \sa @ref apex_stop, @ref apex_resume, @ref apex_yield
 */
APEX_EXPORT apex_profiler_handle apex_start(apex_profiler_type type, void * identifier);

/**
 \brief Stop a timer.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process.
 
 \param profiler The handle of the profiler object.
 \return No return value.
 \sa @ref apex_start, @ref apex_yield, @ref apex_resume
 */
APEX_EXPORT void apex_stop(apex_profiler_handle profiler);

/**
 \brief Stop a timer, but don't increment the number of calls.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process. The number of calls
 will NOT be incremented - this "task" was yielded, not completed.
 It will be resumed by another thread at a later time.
 
 \param profiler The handle of the profiler object.
 \return No return value.
 \sa @ref apex_start, @ref apex_stop, @ref apex_resume
 */
APEX_EXPORT void apex_yield(apex_profiler_handle profiler);

/**
 \brief Resume a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the name
 and/or function address passed in to this function.
 The difference between this function and the apex_start
 function is that the number of calls to that
 timer will not be incremented.
 
 \param type The type of the address to be stored. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be timed, or a "const
             char *" pointer to the name of the timer.
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching @ref apex_stop
         call when the timer should be stopped.
 \sa @ref apex_start, @ref apex_stop, @ref apex_yield
*/
APEX_EXPORT apex_profiler_handle apex_resume(apex_profiler_type type, void * identifier);

/*
 * Functions for resetting timer values
 */

/**
 \brief Reset a timer or counter.

 This function will reset the profile associated with the specified
 timer or counter id to zero.
 
 \param type The type of the address to be reset. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be reset, or a "const
             char *" pointer to the name of the timer / counter.
 \return No return value.
 \sa @ref apex_get_profile
 */
APEX_EXPORT void apex_reset(apex_profiler_type type, void * identifier);

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
 \brief Create a new task (dependency).

 This function will note a task dependency between the current 
 timer (task) and the new task.

 \param type The type of the address to be reset. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function of the task, or a "const
             char *" pointer to the name of the task.
 \param task_id The ID of the task 
 \return No return value.
 */

APEX_EXPORT void apex_new_task(apex_profiler_type type, void * identifier, void * task_id);

/**
 \brief Register an event type with APEX.

 Create a user-defined event type for APEX.
 
 \param name The name of the custom event
 \return The index of the custom event.
 \sa @ref apex_custom_event
 */
APEX_EXPORT apex_event_type apex_register_custom_event(const char * name);

/**
 \brief Trigger a custom event.

 This function will pass a custom event to the APEX event listeners.
 Each listeners' custom event handler will handle the custom event.
 Policy functions will be passed the custom event name in the event context.
 
 \param event_type The type of the custom event
 \param custom_data Data specific to the custom event
 \return No return value.
 \sa @ref apex_register_custom_event
 */
APEX_EXPORT void apex_custom_event(apex_event_type event_type, void * custom_data);

/*
 * Utility functions
 */

/**
 \brief Return the APEX version.
 
 \return A character string with the APEX version. This string
 should not be freed after the calling function is done with it.
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
APEX_EXPORT void apex_set_node_id(int id);

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

/**
 \brief Exit a thread.

 For multithreaded applications, exit this thread and clean up.
 \warning Failure to exit a thread with APEX may invalidate
 statistics.
 
 \return No return value.
 */
APEX_EXPORT void apex_exit_thread(void);

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
 \sa @ref apex_deregister_policy, @ref apex_register_periodic_policy
 */
APEX_EXPORT apex_policy_handle * apex_register_policy(const apex_event_type when, apex_policy_function f);

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 periodically.  This assigns the passed in function to be called on a periodic
 basis.  The context for the event will be passed to the registered function.
 
 \param period How frequently the function should be called
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 \sa @ref apex_deregister_policy, @ref apex_register_policy
 */
APEX_EXPORT apex_policy_handle * apex_register_periodic_policy(unsigned long period, apex_policy_function f);

/**
 \brief Deregister a policy with APEX.

 This function will deregister the specified policy. In order to enable the policy
 again, it should be registered using @ref apex_register_policy or @ref apex_register_periodic_policy.
 
 \param handle The handle of the policy to be deregistered.
 \sa @ref apex_register_policy, @ref apex_register_periodic_policy
 */
APEX_EXPORT void apex_deregister_policy(apex_policy_handle * handle);

/**
 \brief Get the current profile for the specified id.

 This function will return the current profile for the specified profiler id.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date.  This profile can be either a timer or a sampled value.
 
 \param type The type of the address to be returned. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be returned, or a "const
             char *" pointer to the name of the timer / counter.
 \return The current profile for that timed function or sampled value.
 */
APEX_EXPORT apex_profile * apex_get_profile(apex_profiler_type type, void * identifier);

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

 \param type The type of the address to be optimized. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be optimized, or a "const
             char *" pointer to the name of the counter/timer.
 \param criteria The optimization criteria.
 \param method The optimization method.
 \param update_interval The time between observations, in microseconds.
 \return APEX_NOERROR on success, otherwise an error code.
 */

APEX_EXPORT int apex_setup_timer_throttling(apex_profiler_type type, 
        void * identifier,
        apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval);

/**
 \brief Setup throttling to optimize for the specified function, using
        multiple input criteria.

 This function will initialize a policy to optimize the specified function, 
 using the list of tunable inputs for the specified function. The 
 optimization criteria include maximizing throughput,
 minimizing or maximizing time spent in the specified function. After
 evaluating the state of the system, the policy will assign new values to
 the inputs.

 \param type The type of the address to be optimized. This can be one of the @ref
             apex_profiler_type values.
 \param identifier The function address of the function to be optimized, or a "const
             char *" pointer to the name of the counter/timer.
 \param criteria The optimization criteria.
 \param event_type The @ref apex_event_type that should trigger this policy 
 \param num_inputs The number of tunable inputs for optimization
 \param inputs An array of addresses to inputs for optimization
 \param mins An array of minimum values for each input
 \param maxs An array of maximum values for each input
 \param steps An array of step values for each input
 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int apex_setup_throughput_tuning(
        apex_profiler_type type,
        void * identifier,
        apex_optimization_criteria_t criteria,
        apex_event_type event_type, int num_inputs, long ** inputs, long * mins,
        long * maxs, long * steps);

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

/**
 \brief Set the current thread cap for throttling.

 This function will set the current thread cap based on an external throttling
 policy.

 \param new_cap The current thread cap value.
 */
APEX_EXPORT void apex_set_thread_cap(int new_cap);             // for thread throttling

/**
 \brief Print the current APEX settings

 This function will print all the current APEX settings.
 */
APEX_EXPORT void apex_print_options(void);

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define apex_macro(name, member_variable, type, default_value) \
void apex_set_##member_variable (type inval); \
type apex_get_##member_variable (void);
FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#define apex_macro(name, member_variable, type, default_value) \
void apex_set_##member_variable (type inval); \
type apex_get_##member_variable (void);
FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
}
#endif

#endif //APEX_H

