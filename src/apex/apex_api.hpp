//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

/* required for Doxygen */
/** @file */ 
#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#endif

#include "apex_types.h"
#include "apex_options.hpp"
#include "apex_export.h" 
#include "profiler.hpp" 
#include "profile.hpp" 
#include "task_wrapper.hpp" 
#include <functional>
#include <stdio.h>
#include <string>
#include <set>
#include <vector>
#include <stdint.h>

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

class apex_tuning_request;

/**
 \brief The main APEX namespace.
 
 The C++ interface for APEX uses the apex namespace. In comparison,
 The C interface has functions that start with "apex_".

 */
namespace apex
{

// declare a default "null" pointer for std::shared_ptr<task_wrapper>& references
static std::shared_ptr<task_wrapper> null_task_wrapper(nullptr);

// These are all static functions for the class. There should be only
// one APEX object in the process space.

/**
 \brief Intialize APEX.
 \warning For best results, this function should be called before any other 
          APEX functions. 
 \warning Use this version of apex::init when you do not have access
          to the input arguments.
 
 \param thread_name The name of the thread, or NULL. The lifetime of the
                    thread will be timed with a timer using this same name.
 \param comm_rank   The rank of this process within the full distributed
                    application, i.e. MPI rank or HPX locality.
 \param comm_size   The total number of processes within the full distributed
                    application, i.e. MPI comm_size or total number of HPX localities.
 \return APEX_NOERROR on success, or APEX_ERROR on failure.
 \sa @ref apex::init @ref apex::finalize
 */
APEX_EXPORT uint64_t init(const char * thread_name, const uint64_t comm_rank, const uint64_t comm_size);

/**
 \brief Dump output from APEX.
 
 The stop measurement method will terminate all measurement and optionally:
 - print a report to the screen
 - write a profile to disk (if requested)
 - output all other visualization data
 \param reset   Whether to reset all statistics
 \return a string containing the output
 \sa @ref apex::finalize
 */
APEX_EXPORT std::string dump(bool reset);

/**
 \brief Finalize APEX.
 
 The stop measurement method will terminate all measurement and optionally:
 - print a report to the screen
 - write a profile to disk (if requested)
 \return No return value.
 \sa @ref apex::init
 */
APEX_EXPORT void finalize(void);

/**
 \brief Cleanup APEX.
 \warning For best results, this function should be explicitly called 
                    to free all memory allocated by APEX. If not explicitly called from
                    the application or runtime, it will be automatically called when the
                    APEX main singleton object is destructed. apex::finalize will be 
                    automatically called from apex::cleanup if it has not yet been called.
 
 The cleanup method will free all allocated memory for APEX.
 \return No return value.
 \sa @ref apex::init @ref apex::finalize
 */
APEX_EXPORT void cleanup(void);

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
         possible, and passed in to the matching apex::stop()
         call when the timer should be stopped.
 \sa @ref apex::stop, @ref apex::yield, @ref apex::resume
 */
APEX_EXPORT profiler * start(const std::string &timer_name);

/**
 \brief Start a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the 
 address passed in to this function.
 
 \param function_address The address of the function to be timed
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching apex::stop
         call when the timer should be stopped.
 \sa @ref apex::stop, @ref apex::yield, @ref apex::resume
 */
APEX_EXPORT profiler * start(const apex_function_address function_address);

/**
 \brief Start a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the 
 task_wrapper passed in to this function.
 
 \param function_address The address of the function to be timed
 \param apex::task_wrapper A pointer to an apex::task_wrapper created
         by apex::new_task. APEX will use this to store the 
         profiler data.
 \return The handle for the profiler object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching apex::stop
         call when the timer should be stopped.
 \sa @ref apex::stop, @ref apex::yield, @ref apex::resume @ref apex::new_task
 */
APEX_EXPORT profiler * start(std::shared_ptr<task_wrapper> &task_wrapper_ptr);

/**
 \brief Stop a timer.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process.
 
 \param the_profiler The handle of the profiler object.
 \return No return value.
 \sa @ref apex::start, @ref apex::yield, @ref apex::resume
 */
APEX_EXPORT void stop(profiler * the_profiler, bool cleanup=true);

/**
 \brief Stop a timer.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process.
 
 \param task_wrapper_ptr an apex::task_wrapper pointer that was started
 \return No return value.
 \sa @ref apex::start, @ref apex::yield, @ref apex::resume, @ref apex::new_task
 */
APEX_EXPORT void stop(std::shared_ptr<task_wrapper> &task_wrapper_ptr);

/**
 \brief Stop a timer, but don't increment the number of calls.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process. The number of calls
 will NOT be incremented - this "task" was yielded, not completed.
 It will be resumed by another thread at a later time.
 
 \param the_profiler The handle of the profiler object.
 \return No return value.
 \sa @ref apex::start, @ref apex::stop, @ref apex::resume
 */
APEX_EXPORT void yield(profiler * the_profiler);

/**
 \brief Stop a timer, but don't increment the number of calls.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process. The number of calls
 will NOT be incremented - this "task" was yielded, not completed.
 It will be resumed by another thread at a later time.
 
 \param task_wrapper_ptr an apex::task_wrapper pointer that was started
 \return No return value.
 \sa @ref apex::start, @ref apex::stop, @ref apex::resume
 */
APEX_EXPORT void yield(std::shared_ptr<task_wrapper> &task_wrapper_ptr);

/**
 \brief Resume a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the name
 passed in to this function.
 The difference between this function and the apex::start
 function is that the number of calls to that
 timer will not be incremented.
 
 \param timer_name The name of the timer.
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching apex::stop()
         call when the timer should be stopped.
 \sa @ref apex::stop, @ref apex::yield, @ref apex::start
 */
APEX_EXPORT profiler * resume(const std::string &timer_name);

/**
 \brief Resume a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the 
 address passed in to this function.
 The difference between this function and the apex::start
 function is that the number of calls to that
 timer will not be incremented.
 
 \param function_address The address of the function to be timed
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching apex::stop
         call when the timer should be stopped.
 \sa apex::stop, apex::yield, apex::start
 */
APEX_EXPORT profiler * resume(const apex_function_address function_address);

/**
 \brief Resume a timer.

 This function will create a profiler object in APEX, and return a
 handle to the object.  The object will be associated with the 
 address passed in to this function.
 The difference between this function and the apex::start
 function is that the number of calls to that
 timer will not be incremented.
 
 \param function_address The address of the function to be timed
 \param apex::task_wrapper A pointer to an apex::task_wrapper created
         by apex::new_task. APEX will use this to store the 
         profiler data.
 \return The handle for the timer object in APEX. Not intended to be
         queried by the application. Should be retained locally, if
         possible, and passed in to the matching apex::stop
         call when the timer should be stopped.
 \sa apex::stop, apex::yield, apex::start
 */
APEX_EXPORT profiler * resume(std::shared_ptr<task_wrapper> &task_wrapper_ptr);

/*
 * Functions for resetting timer values
 */

/**
 \brief Reset a timer or counter.

 This function will reset the profile associated with the specified
 timer or counter name to zero.
 
 \param timer_name The name of the timer.
 \return No return value.
 \sa @ref apex::get_profile
 */
APEX_EXPORT void reset(const std::string &timer_name);

/**
 \brief Reset a timer.

 This function will reset the profile associated with the specified
 timer to zero.
 
 \param function_address The function address of the timer.
 \return No return value.
 */
APEX_EXPORT void reset(apex_function_address function_address);

/**
 \brief Set the thread state

 This function will set the thread state in APEX for 3rd party observation
 
 \param state The state of the thread.
 \return No return value.
 */
APEX_EXPORT void set_state(apex_thread_state state);

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
APEX_EXPORT void sample_value(const std::string &name, double value);

/**
 \brief Create a new task (dependency).

 This function will note a task dependency between the current 
 timer (task) and the new task.

 \param name The name of the timer.                                        
 \param task_id The ID of the task (default of UINTMAX_MAX implies none provided by runtime)
 \return pointer to an apex::task_wrapper object
 */

APEX_EXPORT std::shared_ptr<task_wrapper> new_task(
    const std::string &name, 
    const uint64_t task_id = UINTMAX_MAX, 
    const std::shared_ptr<apex::task_wrapper> &parent_task = null_task_wrapper);

/**
 \brief Create a new task (dependency).

 This function will note a task dependency between the current 
 timer (task) and the new task.

 \param function_address The function address of the timer.
 \param task_id The ID of the task (default of -1 implies none provided by runtime)
 \return pointer to an apex::task_wrapper object
 */

APEX_EXPORT std::shared_ptr<task_wrapper> new_task(
    const apex_function_address function_address,
    const uint64_t task_id = UINTMAX_MAX,
    const std::shared_ptr<apex::task_wrapper> &parent_task = null_task_wrapper);

/**
 \brief Update a task (dependency).

 This function will update the name that this task wrapper refers to.

 \param wrapper The existing apex::task_wrapper object
 \param name The new name of the timer.
 */

APEX_EXPORT std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> &wrapper, 
    const std::string &name);

/**
 \brief Update a task wrapper (dependency).

 This function will update the function address that this task wrapper refers to.

 \param wrapper The existing apex::task_wrapper object
 \param function_address The new function address of the timer.
 */

APEX_EXPORT std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> &wrapper, 
    const apex_function_address function_address);

/**
 \brief Register an event type with APEX.

 Create a user-defined event type for APEX.
 
 \param name The name of the custom event
 \return The index of the custom event.
 \sa @ref apex::custom_event
 */
APEX_EXPORT apex_event_type register_custom_event(const std::string &name);

/**
 \brief Trigger a custom event.

 This function will pass a custom event to the APEX event listeners.
 Each listeners' custom event handler will handle the custom event.
 Policy functions will be passed the custom event name in the event context.
 
 \param event_type The type of the custom event
 \param custom_data Data specific to the custom event
 \return No return value.
 \sa @ref apex::register_custom_event
 */
APEX_EXPORT void custom_event(apex_event_type event_type, void * custom_data);

/*
 * Utility functions
 */

/**
 \brief Return the APEX version.
 
 \return A string with the APEX version.
 */
APEX_EXPORT std::string& version(void);

/**
 \brief Register a new thread.

 For multithreaded applications, register a new thread with APEX.
 \warning Failure to register a thread with APEX may invalidate
 statistics, and may prevent the ability to use timers or sampled
 values for this thread.
 
 \param name The name that will be assigned to the new thread.
 \return No return value.
 */
APEX_EXPORT void register_thread(const std::string &name);

/**
 \brief Exit a thread.

 For multithreaded applications, exit this thread and clean up.
 \warning Failure to exit a thread with APEX may invalidate
 statistics.
 
 \return No return value.
 */
APEX_EXPORT void exit_thread(void);

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
 \sa @ref apex::deregister_policy, @ref apex::register_periodic_policy
 */
APEX_EXPORT apex_policy_handle* register_policy(const apex_event_type when, std::function<int(apex_context const&)> f);

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 when certain events occur in the APEX library, or periodically.
 This assigns the passed in function to the event, so that when that
 event occurs in APEX, the function is called. The context for the
 event will be passed to the registered function.
 
 \param when The set of APEX events when this function should be called
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 \sa @ref apex::deregister_policy, @ref apex::register_periodic_policy
 */
APEX_EXPORT std::set<apex_policy_handle*> register_policy(std::set<apex_event_type> when, std::function<int(apex_context const&)> f);

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 periodically.  This assigns the passed in function to be called on a periodic
 basis.  The context for the event will be passed to the registered function.
 
 \param period How frequently the function should be called (in microseconds)
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 */
APEX_EXPORT apex_policy_handle* register_periodic_policy(unsigned long period, std::function<int(apex_context const&)> f);

/**
 \brief Periodically sample a runtime counter.

 Apex provides the ability to call an application-specified function
 periodically.  This assigns the passed in function to be called on a periodic
 basis.  The context for the event will be passed to the registered function.
 
 \param period How frequently the counter should be called (in microseconds)
 \param counter_name The name of the counter to sample 
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 */
APEX_EXPORT apex_policy_handle* sample_runtime_counter(unsigned long period, const std::string & counter_name);


/**
 \brief Deregister a policy with APEX.

 This function will deregister the specified policy. In order to enable the policy
 again, it should be registered using @ref apex::register_policy or @ref apex::register_periodic_policy.
   
 \param handle The handle of the policy to be deregistered.
 \sa apex::register_policy, apex::register_periodic_policy
 */
APEX_EXPORT void deregister_policy(apex_policy_handle * handle);

/**
 \brief Get the current profile for the specified function address.

 This function will return the current profile for the specified address.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date. 
 
 \param function_address The address of the function.
 \return The current profile for that timed function.
 */
APEX_EXPORT apex_profile* get_profile(apex_function_address function_address);

/**
 \brief Get the current profile for the specified function address.

 This function will return the current profile for the specified address.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date.  This profile can be either a timer or a sampled value.
 
 \param timer_name The name of the function
 \return The current profile for that timed function or sampled value.
 */
APEX_EXPORT apex_profile* get_profile(const std::string &timer_name);

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 \brief Get the set of profiles that are identified by name
 \internal

 This function will return the names of current profiles for all timers and counters.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date.  This profile can be either a timer or a sampled value.
 
 \return A vector of strings containing the list of names.
 */
/* Disabled, after adding task_identifier support */
//APEX_EXPORT std::vector<std::string> get_available_profiles();

/**
 \brief Get the current power reading

 This function will return the current power level for the node, measured in Watts.

 \return The current power level in Watts.
 */
APEX_EXPORT double current_power_high(void);

#endif // DOXYGEN_SHOULD_SKIP_THIS

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
 which can be queried using @ref apex::get_thread_cap().
 
 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int setup_power_cap_throttling(void);      // initialize

/**
 \brief Setup throttling to optimize for the specified function.

 This function will initialize the throttling policy to optimize for the 
 specified function. The optimization criteria include maximizing throughput,
 minimizing or maximizing time spent in the specified function. After
 evaluating the state of the system, the policy will set the thread cap, which
 can be queried using @ref apex::get_thread_cap().

 \param the_address The address of the function to be optimized.
 \param criteria The optimization criteria.
 \param method The optimization method.
 \param update_interval The time between observations, in microseconds.
 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int setup_timer_throttling(apex_function_address the_address,
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

 \param the_address The address of the function to be optimized.
 \param criteria The optimization criteria.
 \param event_type The @ref apex_event_type that should trigger this policy 
 \param num_inputs The number of tunable inputs for optimization
 \param inputs An array of addresses to inputs for optimization
 \param mins An array of minimum values for each input
 \param maxs An array of maximum values for each input
 \param steps An array of step values for each input
 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int setup_throughput_tuning(apex_function_address the_address,
        apex_optimization_criteria_t criteria,
        apex_event_type event_type, int num_inputs, long ** inputs, long * mins,
        long * maxs, long * steps);

/**
 \brief Setup tuning of specified parameters to optimize for a custom metric, using
        multiple input criteria.

 This function will initialize a policy to optimize a custom metric, 
 using the list of tunable parameters.  
 The system tries to minimize the custom metric.
 After evaluating the state of the system, the policy will assign new values to
 the inputs.

 \param metric A function returning the value to be minimized.
 \param event_type The @ref apex_event_type that should trigger this policy 
 \param num_inputs The number of tunable inputs for optimization
 \param inputs An array of addresses to inputs for optimization
 \param mins An array of minimum values for each input
 \param maxs An array of maximum values for each input
 \param steps An array of step values for each input
 \return A handle to the tuning session
 */
APEX_EXPORT apex_tuning_session_handle setup_custom_tuning(std::function<double(void)> metric, 
        apex_event_type event_type, int num_inputs, long ** inputs, long * mins,
        long * maxs, long * steps);

/**
 \brief Setup tuning of specified parameters to optimize for a custom metric, using
        multiple input criteria of potentially multiple types.

 This function will initialize a policy to optimize a custom metric, 
 using metric and parameters specified in the tuning request.
 The system tries to minimize the custom metric.

 \param request An apex_tuning_request object that specifies the tuning parameters.
 \return A handle to the tuning session.
 */
APEX_EXPORT apex_tuning_session_handle setup_custom_tuning(apex_tuning_request & request);

/**
 \brief Setup throttling to optimize for the specified function or counter.

 This function will initialize the throttling policy to optimize for the 
 specified function or counter. The optimization criteria include maximizing
 throughput, minimizing or maximizing time spent in the specified function
 or value sampled in the counter. After evaluating the state of the system,
 the policy will set the thread cap, which can be queried using 
 @ref apex::get_thread_cap().

 \param the_name The name of the function or counter to be optimized.
 \param criteria The optimization criteria.
 \param method The optimization method.
 \param update_interval The time between observations, in microseconds.

 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int setup_timer_throttling(const std::string &the_name,
        apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
APEX_EXPORT int startup_throttling(void);
#endif

/**
 \brief Terminate the throttling policy.

 This function will terminate the throttling policy.

 \return APEX_NOERROR on success, otherwise an error code.
 */
APEX_EXPORT int shutdown_throttling(void);   // terminate

/**
 \brief Get the current thread cap set by the throttling.

 This function will return the current thread cap based on the throttling
 policy.

 \return The current thread cap value.
 */
APEX_EXPORT int get_thread_cap(void);             // for thread throttling
#ifndef DOXYGEN_SHOULD_SKIP_THIS // test API stuff
APEX_EXPORT int get_input2(void);             // for thread throttling
#endif

/**
 \brief Set the current thread cap for throttling.

 This function will set the current thread cap based on an external throttling
 policy.

 \param new_cap The current thread cap value.
 */
APEX_EXPORT void set_thread_cap(int new_cap);             // for thread throttling


/**
 \brief Return a vector of the current tunable parameters

 \return A vector of pairs; the first element is the name of the tunable parameter,
 while the second is a pointer to its value.

 */
APEX_EXPORT std::vector<std::pair<std::string,long*>> & get_tunable_params(apex_tuning_session_handle h);


/**
 \brief Check whether a tuning session has converged.

 \param handle The handle for the tuning session of interest.

 \return true if the tuning session has converged, otherwise false 

 */
APEX_EXPORT bool has_session_converged(apex_tuning_session_handle handle);


/**
 \brief Set a tuning session's values to the best known values.

 \param h The handle for the tuning session of interest.

 */
APEX_EXPORT void get_best_values(apex_tuning_session_handle h);

/**
 \brief Print out all configuration settings for APEX.

 */
APEX_EXPORT void print_options(void);

/**
 \brief Notify APEX that the current thread is sending a parcel/message
        to another rank/locality/process.

 This method notifies APEX that the current thread is sending a
 parcel/message to another rank/locality/process. The tag is meant
 to be an identifier for the message, not required to be unique.
 The target value is the APEX rank of the target of the message.

 \param tag The message identifier
 \param size The message size (in bytes)
 \param target The message target (as an APEX rank)
 */
APEX_EXPORT void send (uint64_t tag, uint64_t size, uint64_t target);

/**
 \brief Notify APEX that the current thread is receiving a parcel/message
        from another rank/locality/process.

 This method notifies APEX that the current thread is receiving a
 parcel/message from another rank/locality/process. The tag is meant
 to be an identifier for the message, not required to be unique.
 The source value is the APEX rank of the source of the message.

 \param tag The message identifier
 \param size The message size (in bytes)
 \param source_rank The message source (as a rank/locality index)
 \param source_thread The message source (as a worker index - 0 if unknown)
 */
APEX_EXPORT void recv (uint64_t tag, uint64_t size, uint64_t source_rank, uint64_t source_thread);

/**
 \brief A convenience class for using APEX in C++ applications.
 
  This class will automatically start an APEX timer when the object is constructed,
  and automatically stop the timer when the object goes out of scope.
  There are options for registering a thread when creating a timer for the first
  time on a thread.

 */
class scoped_timer {
    private:
        std::shared_ptr<apex::task_wrapper> twp;
    public:
/**
 \brief Construct and start an APEX timer.

 \param func The address of a function used to identify the timer type
 */
        scoped_timer(uint64_t func) : twp(nullptr) {
            twp = apex::new_task((apex_function_address)func);
            apex::start(twp);
        }
/**
 \brief Construct and start an APEX timer.

 \param func The name of a function used to identify the timer type
 */
        scoped_timer(std::string func) : twp(nullptr) {
            twp = apex::new_task(func);
            apex::start(twp);
        }
/**
 \brief Register a new thread with APEX, then construct and start an APEX timer.

 \param func The address of a function used to identify the timer type
 \param thread_name The name of this new worker thread in the runtime
 */
        scoped_timer(uint64_t func, std::shared_ptr<apex::task_wrapper> &parent)
            : twp(nullptr) {
            twp = apex::new_task((apex_function_address)func, UINTMAX_MAX, parent);
            apex::start(twp);
        }
/**
 \brief Register a new thread with APEX, then construct and start an APEX timer.

 \param func The name of a function used to identify the timer type
 \param thread_name The name of this new worker thread in the runtime
 */
        scoped_timer(std::string func, std::shared_ptr<apex::task_wrapper> &parent)
            : twp(nullptr) {
            twp = apex::new_task(func, UINTMAX_MAX, parent);
            apex::start(twp);
        }
/**
 \brief Stop the APEX timer.

 */
        void stop(void) {
            if (twp != nullptr) { 
                apex::stop(twp); 
                twp = nullptr; 
            } 
 /**
 \brief Destructor.

 */
       }
        ~scoped_timer() { 
            stop();
        }

/*
 \brief Get the internal task wrapper object
 */
        std::shared_ptr<apex::task_wrapper>& get_task_wrapper(void) {
            return twp;
        }
};

class scoped_thread {
public:
    scoped_thread(const std::string& thread_name) {
        apex::register_thread(thread_name);
    }
    ~scoped_thread() {
        apex::exit_thread();
    }
};

/**
 \brief A convenient macro for inserting an APEX self-stopping timer.

 This macro will create a timer using the values of __func__, __LINE__ and
 __FILE__ from the preprocessor.

 */
#define APEX_SCOPED_TIMER \
      std::ostringstream _s_foo; \
      _s_foo << __func__ << " [" << __FILE__ << ":" << __LINE__ << "]"; \
      _name_foo = std::string(_s_foo.str()); \
    scoped_timer __foo(_name_foo); 

} //namespace apex

