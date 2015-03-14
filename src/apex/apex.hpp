//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_HPP
#define APEX_HPP

/* required for Doxygen */
/** @file */ 
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#include <hpx/include/runtime.hpp>
#endif

#include <string>
#include <vector>
#include <stdint.h>
#include "apex_types.h"
#include "handler.hpp"
#include "event_listener.hpp"
#include "policy_handler.hpp"
#include "profiler_listener.hpp"
#include "apex_options.hpp"
#include "apex_export.h" 

#ifdef APEX_HAVE_RCR
#include "libenergy.h"
#endif

#if APEX_HAVE_PROC
#include "proc_read.h" 
#endif

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 \brief The main APEX namespace.
 
 The C++ interface for APEX uses the apex namespace. In comparison,
 The C interface has functions that start with "apex_".

 */
namespace apex
{

///////////////////////////////////////////////////////////////////////
// Main class for the APEX project

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/*
 The APEX class is only instantiated once per process (i.e. it is a
 singleton object). The instance itself is only used internally. The
 C++ interface for APEX uses the apex namespace. The C interface
 has functions that start with "apex_".
 */
class apex
{
private:
// private constructors cannot be called
    apex() : m_argc(0), m_argv(NULL), m_node_id(0), m_my_locality(new std::string("0"))
    {
        _initialize();
    };
    apex(int argc, char**argv) : m_argc(argc), m_argv(argv), m_node_id(0), m_my_locality(new std::string("0"))
    {
        _initialize();
    };
    apex(apex const&);            // copy constructor is private
    apex& operator=(apex const& a); // assignment operator is private
// member variables
    static apex* m_pInstance;
    int m_argc;
    char** m_argv;
    int m_node_id;
    bool m_profiling;
    void _initialize();
    policy_handler * m_policy_handler;
    std::map<int, policy_handler*> period_handlers;
    std::vector<apex_thread_state> thread_states;
#ifdef APEX_HAVE_HPX3
    hpx::runtime * m_hpx_runtime;
#endif
public:
    std::vector<event_listener*> listeners;
    std::string* m_my_locality;
    static apex* instance(); // singleton instance
    static apex* instance(int argc, char** argv); // singleton instance
    void set_node_id(int id);
    int get_node_id(void);
#ifdef APEX_HAVE_HPX3
    void set_hpx_runtime(hpx::runtime * hpx_runtime);
    hpx::runtime * get_hpx_runtime(void);
#endif
    //void notify_listeners(event_data* event_data_);
    policy_handler * get_policy_handler(void) const;
/*
    template <typename Rep, typename Period>
    policy_handler * get_policy_handler(std::chrono::duration<Rep, Period> const& period);
*/
    policy_handler * get_policy_handler(uint64_t const& period_microseconds);
    void set_state(int thread_id, apex_thread_state state) { thread_states[thread_id] = state; }
    apex_thread_state get_state(int thread_id) { return thread_states[thread_id]; }
    void resize_state(int thread_id) { 
        static boost::mutex _mtx;
        _mtx.lock();
        if ((unsigned int)thread_id >= thread_states.size()) {
            thread_states.resize(thread_id + 1); 
        }
        _mtx.unlock();
    }
    ~apex();
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

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
 \return No return value.
 \sa apex::init
 \sa apex::finalize
 */
APEX_EXPORT void init(const char * thread_name);

/**
 \brief Intialize APEX.
 \warning For best results, this function should be called before any other 
          APEX functions. 
 \warning Use this version of apex::init when you have access
          to the input arguments.
 
 \param argc The number of arguments passed in to the program.
 \param argv An array of arguments passed in to the program.
 \param thread_name The name of the thread, or NULL. The lifetime of the
                    thread will be timed with a timer using this same name.
 \return No return value.
 \sa apex::init
 \sa apex::finalize
 */
APEX_EXPORT void init(int argc, char** argv, const char * thread_name);

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
 \sa apex::init
 */
APEX_EXPORT void finalize(void);

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
 \sa apex::stop
 */
APEX_EXPORT profiler* start(const std::string &timer_name);

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
 \sa apex::stop
 */
APEX_EXPORT profiler* start(apex_function_address function_address);

/**
 \brief Stop a timer.

 This function will stop the specified profiler object, and queue
 the profiler to be processed out-of-band. The timer value will 
 eventually added to the profile for the process.
 
 \param the_profiler The handle of the profiler object.
 \return No return value.
 \sa apex::start
 */
APEX_EXPORT void stop(profiler* the_profiler);

/**
 \brief Restart a timer.

 This function will restart the specified profiler object. The
 difference between this function and the apex::start 
 function is that the number of calls to that
 timer will not be incremented.
 
 \param the_profiler The handle of the profiler object.
 \return No return value.
 \sa apex::start, apex::stop
 */
APEX_EXPORT void resume(profiler* the_profiler);

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
 \brief Trigger a custom event.

 This function will pass a custom event to the APEX event listeners.
 Each listeners' on_custom_event() event will handle the custom event.
 Policy functions will be passed the custom event name in the event context.
 
 \param event_name The name of the custom event
 \param custom_data Data relevant to the custom event
 \return No return value.
 */
APEX_EXPORT void custom_event(const std::string &event_name, void * custom_data);

/**
 \brief Reset a timer or counter.

 This function will reset the profile associated with the specified
 timer or counter name to zero.
 
 \param timer_name The name of the timer.
 \return No return value.
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

/**
 \brief Return the APEX version.
 
 \return A string with the APEX version.
 */
APEX_EXPORT std::string version(void);

/**
 \brief Set this process' node ID.

 For distributed applications, this function will store the
 node ID. Common values are the MPI rank, the HPX locality, etc.
 This ID will be used to identify the process in the global
 performance space.
 
 \param id The node ID for this process.
 \return No return value.
 */
APEX_EXPORT void set_node_id(int id);

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

#ifndef DOXYGEN_SHOULD_SKIP_THIS // not sure if these will stay in the API

APEX_EXPORT void track_power(void);
APEX_EXPORT void track_power_here(void);
APEX_EXPORT void enable_tracking_power(void);
APEX_EXPORT void disable_tracking_power(void);
APEX_EXPORT void set_interrupt_interval(int seconds);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

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
APEX_EXPORT apex_policy_handle* register_policy(const apex_event_type when, std::function<int(apex_context const&)> f);

/**
 \brief Register a policy with APEX.

 Apex provides the ability to call an application-specified function
 periodically.  This assigns the passed in function to be called on a periodic
 basis.  The context for the event will be passed to the registered function.
 
 \param period How frequently the function should be called
 \param f The function to be called when that event is handled by APEX.
 \return A handle to the policy, to be stored if the policy is to be un-registered later.
 */
APEX_EXPORT apex_policy_handle* register_periodic_policy(unsigned long period, std::function<int(apex_context const&)> f);

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

/**
 \brief Get the set of profiles that are identified by name

 This function will return the names of current profiles for all timers and counters.
 Because profiles are updated out-of-band, it is possible that this profile
 value is out of date.  This profile can be either a timer or a sampled value.
 
 \return A vector of strings containing the list of names.
 */
APEX_EXPORT std::vector<std::string> get_available_profiles();

/**
 \brief Get the current power reading

 This function will return the current power level for the node, measured in Watts.

 \return The current power level in Watts.
 */
APEX_EXPORT inline double current_power_high(void) {
#ifdef APEX_HAVE_RCR
  return (double)rcr_current_power_high();
#elif APEX_HAVE_PROC
  return (double)read_power();
#else
  return 0.0;
#endif
}

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

#ifdef APEX_HAVE_HPX3
hpx::runtime * get_hpx_runtime_ptr(void);
#endif
} //namespace apex

#endif //APEX_HPP
