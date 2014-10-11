//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "apex.hpp"
#include "apex_config.h"
#ifdef APEX_HAVE_RCR
#include "energy_stat.h"
#endif
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

__thread bool _registered = false;
static bool _initialized = false;
__thread int _level = -1;

#if 0
#define APEX_TRACER {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = TAU_PROFILE_GET_THREAD(); \
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
 for (int i = 0 ; i < _level ; i++) \
	ss << "  "; \
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

/** The destructor will request power data from RCRToolkit
 **/
apex::~apex()
{
    APEX_TRACER
#ifdef APEX_HAVE_RCR
    cout << "Getting energy..." << endl;
    energy_daemon_term();
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

/*
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void apex::_initialize()
{
    APEX_TRACER
    this->m_pInstance = this;
    this->m_policy_handler = nullptr;
#ifdef APEX_HAVE_RCR
    uint64_t waitTime = 1000000000L; // in nanoseconds, for nanosleep
    energy_daemon_init(waitTime);
#endif
    char* option = NULL;
#ifdef APEX_HAVE_TAU
    option = getenv("APEX_TAU");
    if (option != NULL)
    {
        listeners.push_back(new tau_listener());
    }
#else
    listeners.push_back(new profiler_listener());
#endif
    option = getenv("APEX_POLICY");
    if (option != NULL)
    {
        this->m_policy_handler = new policy_handler();
        listeners.push_back(this->m_policy_handler);
    }
    option = getenv("APEX_CONCURRENCY");
    if (option != NULL && atoi(option) > 0)
    {
        char* option2 = getenv("APEX_CONCURRENCY_PERIOD");
        if (option2 != NULL)
        {
            listeners.push_back(new concurrency_handler(atoi(option2), option));
        }
        else
        {
            listeners.push_back(new concurrency_handler(option));
        }
    }
}

/** This function is called to create an instance of the class.
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

/*
template <typename Rep, typename Period>
policy_handler * apex::get_policy_handler(std::chrono::duration<Rep, Period> const& period)
*/
policy_handler * apex::get_policy_handler(uint64_t const& period)
{
    char * option = getenv("APEX_POLICY");
    if(option != nullptr && period_handlers.count(period) == 0)
    {
        period_handlers[period] = new policy_handler(period);
    }
    return period_handlers[period];
}

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
    set_node_id(0);
#if HAVE_TAU
    // start top-level timers for threads
    if (thread_name) {
      start(thread_name);
    } else {
      start("APEX MAIN THREAD");
    }
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
    set_node_id(0);
#ifdef APEX_HAVE_TAU
    // start top-level timers for threads
    if (thread_name) {
      start(thread_name);
    } else {
      start("APEX MAIN THREAD");
    }
#endif
}

double version()
{
    APEX_TRACER
    return APEX_VERSION_MAJOR + (APEX_VERSION_MINOR/10.0);
}

void* start(string timer_name)
{
    APEX_TIMER_TRACER("start ", timer_name)
    _level++;
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return NULL; // protect against calls after finalization
    timer_event_data event_data(START_EVENT, thread_instance::get_id(), timer_name);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_start(event_data);
        }
    }
    thread_instance::instance().current_timer = event_data.my_profiler;
    return (void*)(event_data.my_profiler);
}

void* resume(string timer_name)
{
    APEX_TIMER_TRACER("resume", timer_name)
    _level++;
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return NULL; // protect against calls after finalization
    timer_event_data event_data(START_EVENT, thread_instance::get_id(), timer_name);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_resume(event_data);
        }
    }
    thread_instance::instance().current_timer = event_data.my_profiler;
    return (void*)(event_data.my_profiler);
}

void* start(void * function_address) {
    APEX_TIMER_TRACER("start ", timer_name)
    _level++;
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return NULL; // protect against calls after finalization
    timer_event_data event_data(START_EVENT, thread_instance::get_id(), function_address);
    //event_data.function_address = function_address;
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_start(event_data);
        }
    }
    thread_instance::instance().current_timer = event_data.my_profiler;
    return (void*)(event_data.my_profiler);
}

/*
void* resume(void * function_address) {
    return resume(thread_instance::instance().map_addr_to_name(function_address));
}
*/

void resume(void * the_profiler) {
    APEX_TIMER_TRACER("resume", timer_name)
    _level++;
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    timer_event_data event_data(START_EVENT, thread_instance::get_id(), NULL);
    event_data.my_profiler = (profiler *)(the_profiler);
    //instance->notify_listeners(event_data);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_resume(event_data);
        }
    }
    thread_instance::instance().current_timer = event_data.my_profiler;
}

#if 0
void stop(string timer_name)
{
    _level--;
    APEX_TIMER_TRACER("stop  ", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    event_data event_data(STOP_EVENT, thread_instance::get_id(), timer_name);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(event_data);
        }
    }
}

void stop()
{
    _level--;
    APEX_TIMER_TRACER("stop", "?")
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    string empty = "";
    event_data* event_data(STOP_EVENT, thread_instance::get_id(), empty);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(event_data);
        }
    }
}

void stop(void * function_address) {
    stop(thread_instance::instance().map_addr_to_name(function_address));
}
#else
void stop(void * the_profiler)
{
    _level--;
    APEX_TIMER_TRACER("stop  ", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    timer_event_data event_data(STOP_EVENT, thread_instance::get_id(), 0L);
    if (the_profiler == NULL) {
        event_data.my_profiler = thread_instance::instance().current_timer;
    } else {
        event_data.my_profiler = (profiler*)the_profiler;
    }
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(event_data);
        }
    }
}

#endif

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
#endif
}

void finalize()
{
    APEX_TRACER
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

apex_policy_handle* register_policy(const apex_event_type & when,
                    std::function<bool(apex_context const&)> f)
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

/*
template <typename Rep, typename Period>
int register_policy(std::chrono::duration<Rep, Period> const& period,
                    std::function<bool(apex_context const&)> f)
*/
apex_policy_handle* register_periodic_policy(unsigned long period_microseconds,
                    std::function<bool(apex_context const&)> f)
{
    APEX_TRACER
    int id = -1;
    policy_handler * handler = apex::instance()->get_policy_handler(period_microseconds);
    if(handler != nullptr)
    {
        id = handler->register_policy(PERIODIC, f);
    }
    apex_policy_handle * handle = new apex_policy_handle();
    handle->id = id;
    return handle;
}

} // apex namespace

using namespace apex;

extern "C" {

    void apex_init(const char * thread_name)
    {
        APEX_TRACER
        init(thread_name);
    }

    void apex_init_args(int argc, char** argv, const char * thread_name)
    {
        APEX_TRACER
        init(argc, argv, thread_name);
    }

    void apex_finalize()
    {
        APEX_TRACER
        finalize();
    }

    double apex_version()
    {
        APEX_TRACER
        return version();
    }

    void* apex_start(const char * timer_name)
    {
        APEX_TRACER
	if (timer_name)
          return start(string(timer_name));
	else
          return start(string(""));
    }

    void* apex_start_addr(void * function_address)
    {
        APEX_TRACER
        return start(function_address);
    }

    /*
    void apex_resume(const char * timer_name)
    {
        APEX_TRACER
        resume(string(timer_name));
    }

    void apex_resume_addr(void * function_address)
    {
        APEX_TRACER
        resume(function_address);
    }
    */
    void apex_resume_profiler(void * the_profiler)
    {
        APEX_TRACER
        resume(the_profiler);
    }

    /*
    void apex_stop(const char * timer_name)
    {
        APEX_TRACER
	if (timer_name == NULL) {
            stop();
	} else {
            stop(string(timer_name));
	}
    }

    void apex_stop_addr(void * function_address)
    {
        APEX_TRACER
        stop(function_address);
    }
    */
    void apex_stop_profiler(void * the_profiler)
    {
        APEX_TRACER
        stop(the_profiler);
    }

    void apex_sample_value(const char * name, double value)
    {
        APEX_TRACER
        sample_value(string(name), value);
    }

    void apex_set_node_id(int id)
    {
        APEX_TRACER
        set_node_id(id);
    }

    void apex_register_thread(const char * name)
    {
        APEX_TRACER
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


} // extern "C"

