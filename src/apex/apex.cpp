//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "apex.hpp"
#include "apex_types.h"
#include "apex_config.h"
#include "apex_version.h"
#include <iostream>
#include <stdlib.h>
#include <string>
//#include <cxxabi.h> // this is for demangling strings.

#include "concurrency_handler.hpp"
#include "policy_handler.hpp"
#include "thread_instance.hpp"

#ifdef APEX_HAVE_TAU
#include "tau_listener.hpp"
#define PROFILING_ON
//#define TAU_GNU
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#else
#include "profiler_listener.hpp"
#endif

APEX_NATIVE_TLS bool _registered = false;
static bool _initialized = false;

#if 0
#define APEX_TRACER {int __nid = apex::instance()->get_node_id(); \
 int __tid = thread_instance::get_id(); \
 std::stringstream ss; \
 ss << __nid << ":" << __tid << " " << __FUNCTION__ << " ["<< __FILE__ << ":" << __LINE__ << "]" << endl; \
 cout << ss.str();}
#else
#define APEX_TRACER
#endif

#if 0
#define APEX_TIMER_TRACER(A, B) {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = TAU_PROFILE_GET_THREAD(); \
 std::stringstream ss; \
 ss << __nid << ":" << __tid << " " ; \
 ss << (A) << " "<< (B) << endl;\
 cout << ss.str();}
#else
#define APEX_TIMER_TRACER(A, B)
#endif

using namespace std;

namespace apex
{

// Global static pointer used to ensure a single instance of the class.
apex* apex::m_pInstance = NULL;

static bool _notify_listeners = true;
static bool _finalized = false;

#if APEX_HAVE_PROC
    boost::thread * proc_reader_thread;
#endif

/*
 * The destructor will request power data from RCRToolkit
 */
apex::~apex()
{
    APEX_TRACER
#ifdef APEX_HAVE_RCR
    //cout << "Getting energy..." << endl;
    energyDaemonTerm();
#endif
    m_pInstance = NULL;
}

void apex::set_node_id(int id)
{
    APEX_TRACER
    m_node_id = id;
    stringstream ss;
    ss << "locality#" << m_node_id;
    m_my_locality = new string(ss.str());
    node_event_data event_data(id, thread_instance::get_id());
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < listeners.size() ; i++) {
            listeners[i]->on_new_node(event_data);
        }
    }
}

int apex::get_node_id()
{
    APEX_TRACER
    return m_node_id;
}

#ifdef APEX_HAVE_HPX3
static void init_hpx_runtime_ptr(void) {
    apex * instance = apex::instance();
    hpx::runtime * runtime = hpx::get_runtime_ptr();
    instance->set_hpx_runtime(runtime);
}
#endif

/*
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void apex::_initialize()
{
    APEX_TRACER
    this->m_pInstance = this;
    this->m_policy_handler = nullptr;
#ifdef APEX_HAVE_HPX3
    this->m_hpx_runtime = nullptr;
    hpx::register_startup_function(init_hpx_runtime_ptr);
#endif
#ifdef APEX_HAVE_RCR
    uint64_t waitTime = 1000000000L; // in nanoseconds, for nanosleep
    energyDaemonInit();
#endif
    listeners.push_back(new profiler_listener());
#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau())
    {
        listeners.push_back(new tau_listener());
    }
#endif
    if (apex_options::use_policy())
    {
        this->m_policy_handler = new policy_handler();
        listeners.push_back(this->m_policy_handler);
    }
    if (apex_options::use_concurrency() > 0)
    {
        listeners.push_back(new concurrency_handler(apex_options::concurrency_period(), apex_options::use_concurrency()));
    }
#if APEX_HAVE_PROC
    proc_reader_thread = new boost::thread(ProcData::read_proc);
#endif
}

/*  
    This function is called to create an instance of the class.
    Calling the constructor publicly is not allowed. The constructor
    is private and is only called by this Instance function.
*/
apex* apex::instance()
{
    //APEX_TRACER
    // Only allow one instance of class to be generated.
    if (m_pInstance == NULL && !_finalized)
    {
        m_pInstance = new apex;
    }
    return m_pInstance;
}

apex* apex::instance(int argc, char**argv)
{
    //APEX_TRACER
    // Only allow one instance of class to be generated.
    if (m_pInstance == NULL && !_finalized)
    {
        m_pInstance = new apex(argc, argv);
    }
    return m_pInstance;
}

policy_handler * apex::get_policy_handler(void) const
{
    return this->m_policy_handler;
}

policy_handler * apex::get_policy_handler(uint64_t const& period)
{
    if(apex_options::use_policy() && period_handlers.count(period) == 0)
    {
        period_handlers[period] = new policy_handler(period);
        listeners.push_back(period_handlers[period]);
    }
    return period_handlers[period];
}

#ifdef APEX_HAVE_HPX3
void apex::set_hpx_runtime(hpx::runtime * hpx_runtime) {
    m_hpx_runtime = hpx_runtime;
}

hpx::runtime * apex::get_hpx_runtime(void) {
    return m_hpx_runtime;
}
#endif

void init(const char * thread_name)
{
    APEX_TRACER
    if (_registered || _initialized) return; // protect against multiple initializations
    _registered = true;
    _initialized = true;
    int argc = 1;
    const char *dummy = "APEX Application";
    char* argv[1];
    argv[0] = const_cast<char*>(dummy);
    apex* instance = apex::instance(); // get/create the Apex static instance
    if (!instance) return; // protect against calls after finalization
    startup_event_data event_data(argc, argv);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(event_data);
        }
    }
#if HAVE_TAU
    // start top-level timers for threads
    if (thread_name) {
      start(thread_name);
    } else {
      start("APEX MAIN THREAD");
    }
#else 
	APEX_UNUSED(thread_name);
#endif
}

void init(int argc, char** argv, const char * thread_name)
{
    APEX_TRACER
    if (_registered || _initialized) return; // protect against multiple initializations
    _registered = true;
    _initialized = true;
    apex* instance = apex::instance(argc, argv); // get/create the Apex static instance
    if (!instance) return; // protect against calls after finalization
    startup_event_data event_data(argc, argv);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(event_data);
        }
    }
#ifdef APEX_HAVE_TAU
    // start top-level timers for threads
    if (thread_name) {
      start(thread_name);
    } else {
      start("APEX MAIN THREAD");
    }
#else 
	APEX_UNUSED(thread_name);
#endif
}

string version()
{
    APEX_TRACER
    stringstream tmp;
    tmp << APEX_VERSION_MAJOR + (APEX_VERSION_MINOR/10.0);
#if defined (GIT_COMMIT_HASH)
    tmp << "-" << GIT_COMMIT_HASH ;
#endif
#if defined (GIT_BRANCH)
    tmp << "-" << GIT_BRANCH ;
#endif
    tmp << endl;
    return tmp.str().c_str();
}

profiler* start(string timer_name)
{
    APEX_TIMER_TRACER("start ", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return NULL; // protect against calls after finalization
    if (_notify_listeners) {
    string * tmp = new string(timer_name);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_start(0L, tmp);
        }
    }
    return thread_instance::instance().current_timer;
}

profiler* start(apex_function_address function_address) {
    APEX_TIMER_TRACER("start ", function_address)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return NULL; // protect against calls after finalization
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_start(function_address, NULL);
        }
    }
    return thread_instance::instance().current_timer;
}

void reset(std::string timer_name) {
    APEX_TIMER_TRACER("reset", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    if (_notify_listeners) {
        string * tmp = new string(timer_name);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->reset(0L, tmp);
        }
    }
}

void reset(apex_function_address function_address) {
    APEX_TIMER_TRACER("reset", function_address)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->reset(function_address, NULL);
        }
    }
}

void resume(profiler* the_profiler) {
    APEX_TIMER_TRACER("resume", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    thread_instance::instance().current_timer = (profiler *)(the_profiler);
    //instance->notify_listeners(event_data);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_resume((profiler *)(the_profiler));
        }
    }
}

void stop(profiler* the_profiler)
{
    APEX_TIMER_TRACER("stop  ", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    profiler * p;
    if (the_profiler == NULL) {
        p = thread_instance::instance().current_timer;
    } else {
        p = (profiler*)the_profiler;
    }
    if (p == NULL) return;
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(p);
        }
    }
    thread_instance::instance().current_timer = NULL;
}

void sample_value(string name, double value)
{
    APEX_TRACER
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    // parse the counter name
    // either /threadqueue{locality#0/total}/length
    // or     /threadqueue{locality#0/worker-thread#0}/length
    sample_value_event_data* event_data = NULL;
    if (name.find(*(instance->m_my_locality)) != name.npos)
    {
        if (name.find("worker-thread") != name.npos)
        {
            string tmp_name = string(name.c_str());
            // tokenize by / character
            char* token = strtok((char*)tmp_name.c_str(), "/");
            while (token!=NULL) {
              if (strstr(token, "worker-thread")==NULL)
              {
                break;
              }
              token = strtok(NULL, "/");
            }
            int tid = 0;
            if (token != NULL) {
              // strip the trailing close bracket
              token = strtok(token, "}");
              tid = thread_instance::map_name_to_id(token);
            }
            if (tid != -1)
            {
                event_data = new sample_value_event_data(tid, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, tid);
            }
            else
            {
                event_data = new sample_value_event_data(0, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
            }
        }
        else
        {
            event_data = new sample_value_event_data(0, name, value);
            //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
        }
    }
    else
    {
        // what if it doesn't?
        event_data = new sample_value_event_data(0, name, value);
    }
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_sample_value(*event_data);
        }
    }
    delete(event_data);
}

void set_node_id(int id)
{
    APEX_TRACER
    apex* instance = apex::instance();
    if (!instance) return; // protect against calls after finalization
    instance->set_node_id(id);
}

#ifdef APEX_HAVE_HPX3
hpx::runtime * get_hpx_runtime_ptr(void) {
    APEX_TRACER
    apex * instance = apex::instance();
    if (!instance) {
        return nullptr;
    }
    hpx::runtime * runtime = instance->get_hpx_runtime();
    return runtime;
}
#endif

void track_power(void)
{
    APEX_TRACER
#ifdef APEX_HAVE_TAU
    TAU_TRACK_POWER();
#endif
}

void track_power_here(void)
{
    APEX_TRACER
#ifdef APEX_HAVE_TAU
    TAU_TRACK_POWER_HERE();
#endif
}

void enable_tracking_power(void)
{
    APEX_TRACER
#ifdef APEX_HAVE_TAU
    TAU_ENABLE_TRACKING_POWER();
#endif
}

void disable_tracking_power(void)
{
    APEX_TRACER
#ifdef APEX_HAVE_TAU
    TAU_DISABLE_TRACKING_POWER();
#endif
}

void set_interrupt_interval(int seconds)
{
    APEX_TRACER
#ifdef APEX_HAVE_TAU
    TAU_SET_INTERRUPT_INTERVAL(seconds);
#else 
	APEX_UNUSED(seconds);
#endif
}

void finalize()
{
    APEX_TRACER
#if APEX_HAVE_PROC
    ProcData::stop_reading();
#endif
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    if (!_finalized)
    {
        _finalized = true;
        stringstream ss;
        ss << instance->get_node_id();
        shutdown_event_data event_data(instance->get_node_id(), thread_instance::get_id());
        if (_notify_listeners) {
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_shutdown(event_data);
            }
        }
        _notify_listeners = false;
    }
    instance->~apex();
}

void register_thread(string name)
{
    APEX_TRACER
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    if (_registered) return; // protect against multiple registrations on the same thread
    thread_instance::set_name(name);
    new_thread_event_data event_data(name);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_thread(event_data);
        }
    }
#ifdef APEX_HAVE_TAU
    // start top-level timers for threads
    string::size_type index = name.find("#");
    if (index!=std::string::npos)
    {
        string short_name = name.substr(0,index);
        start(short_name);
    }
    else
    {
        start(name);
    }
#endif
}

apex_policy_handle* register_policy(const apex_event_type when,
                    std::function<int(apex_context const&)> f)
{
    APEX_TRACER
    int id = -1;
    policy_handler * handler = apex::instance()->get_policy_handler();
    if(handler != nullptr)
    {
        id = handler->register_policy(when, f);
    }
    apex_policy_handle * handle = new apex_policy_handle();
    handle->id = id;
    return handle;
}

/* How to do it with a chrono object. */

/*
template <typename Rep, typename Period>
int register_policy(std::chrono::duration<Rep, Period> const& period,
                    std::function<int(apex_context const&)> f)
*/

apex_policy_handle* register_periodic_policy(unsigned long period_microseconds,
                    std::function<int(apex_context const&)> f)
 {
    APEX_TRACER
    int id = -1;
    policy_handler * handler = apex::instance()->get_policy_handler(period_microseconds);
    if(handler != nullptr)
    {
        id = handler->register_policy(APEX_PERIODIC, f);
    }
    apex_policy_handle * handle = new apex_policy_handle();
    handle->id = id;
    return handle;
}

apex_profile* get_profile(apex_function_address action_address) {
    profile * tmp = profiler_listener::get_profile(action_address);
    if (tmp != NULL)
        return tmp->get_profile();
    return NULL;
}

apex_profile* get_profile(string &timer_name) {
    profile * tmp = profiler_listener::get_profile(timer_name);
    if (tmp != NULL)
        return tmp->get_profile();
    return NULL;
}

std::vector<std::string> get_available_profiles() {
    return profiler_listener::get_available_profiles();
}

} // apex namespace

using namespace apex;

extern "C" {

    void apex_init(const char * thread_name)
    {
        init(thread_name);
    }

    void apex_init_args(int argc, char** argv, const char * thread_name)
    {
        init(argc, argv, thread_name);
    }

    void apex_finalize()
    {
        finalize();
    }

    const char * apex_version()
    {
        return version().c_str();
    }

    apex_profiler_handle apex_start_name(const char * timer_name)
    {
        if (timer_name)
            return (apex_profiler_handle)start(string(timer_name));
        else
            return (apex_profiler_handle)start(string(""));
    }

    apex_profiler_handle apex_start_address(apex_function_address function_address)
    {
        return (apex_profiler_handle)start(function_address);
    }

    void apex_reset_name(const char * timer_name) {
        if (timer_name) {
            reset(string(timer_name));       
        } else {
            reset(string(""));
        }
    }
    
    void apex_reset_address(apex_function_address function_address) {
        reset(function_address);
    }

    void apex_resume_profiler(apex_profiler_handle the_profiler)
    {
        resume((profiler*)the_profiler);
    }

    void apex_stop_profiler(apex_profiler_handle the_profiler)
    {
        stop((profiler*)the_profiler);
    }

    void apex_sample_value(const char * name, double value)
    {
        sample_value(string(name), value);
    }

    void apex_set_node_id(int id)
    {
        set_node_id(id);
    }

    void apex_register_thread(const char * name)
    {
        if (name) {
            register_thread(string(name));
        } else {
            register_thread(string("APEX WORKER THREAD"));
        }
    }

    void apex_track_power(void)
    {
        track_power();
    }

    void apex_track_power_here(void)
    {
        track_power_here();
    }

    void apex_enable_tracking_power(void)
    {
        enable_tracking_power();
    }

    void apex_disable_tracking_power(void)
    {
        disable_tracking_power();
    }

    void apex_set_interrupt_interval(int seconds)
    {
        set_interrupt_interval(seconds);
    }

    apex_policy_handle* apex_register_policy(const apex_event_type when, int (f)(apex_context const)) {
        return register_policy(when, f);
    }

    apex_policy_handle* apex_register_periodic_policy(unsigned long period, int (f)(apex_context const)) {
        return register_periodic_policy(period, f);
    }

    apex_profile* apex_get_profile_from_address(apex_function_address function_address) {
        return get_profile(function_address);
    }

    apex_profile* apex_get_profile_from_name(const char * timer_name) {
        string tmp(timer_name);
        return get_profile(tmp);
    }

    double apex_current_power_high() {
        return current_power_high();
    }

} // extern "C"


