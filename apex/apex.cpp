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
//#include <cxxabi.h>

#include "concurrency_handler.hpp"
#include "policy_handler.hpp"
#include "tau_listener.hpp"
#include "thread_instance.hpp"

#ifdef APEX_HAVE_TAU
#define PROFILING_ON
//#define TAU_GNU
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif

#if 0
#define APEX_TRACER {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = thread_instance::getID(); \
cout << __nid << ":" << __tid << " " << __FUNCTION__ << " ["<< __FILE__ << ":" << __LINE__ << "]" << endl;}
#else
#define APEX_TRACER
#endif

#if 0
#define APEX_TIMER_TRACER(A, B) {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = TAU_PROFILE_GET_THREAD(); \
cout << __nid << ":" << __tid << " " << (A) << " "<< (B) << endl;}
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
    node_event_data* event_data = new node_event_data(0, id);
    this->notify_listeners(event_data);
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
    set_node_id(0);
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

void apex::notify_listeners(event_data* event_data_)
{
    if (_notify_listeners)
    {
        for (unsigned int i = 0 ; i < listeners.size() ; i++)
        {
            listeners[i]->on_event(event_data_);
        }
    }
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

void init()
{
    APEX_TRACER
    int argc = 1;
    const char *dummy = "APEX Application";
    char* argv[1];
    argv[0] = const_cast<char*>(dummy);
    apex* instance = apex::instance(); // get/create the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_PROFILE_INIT(argc, argv);
    startup_event_data* event_data_ = new startup_event_data(argc, argv);
    instance->notify_listeners(event_data_);
    start("APEX THREAD MAIN");
}

void init(int argc, char** argv)
{
    APEX_TRACER
    apex* instance = apex::instance(argc, argv); // get/create the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_PROFILE_INIT(argc, argv);
    startup_event_data* event_data_ = new startup_event_data(argc, argv);
    instance->notify_listeners(event_data_);
    start("APEX THREAD MAIN");
}

double version()
{
    APEX_TRACER
    return APEX_VERSION_MAJOR + (APEX_VERSION_MINOR/10.0);
}

void start(string timer_name)
{
    APEX_TIMER_TRACER("start", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_START(timer_name.c_str());
    event_data* event_data_ = NULL;
    // don't do this now
    /*
    string* demangled = NULL;
    demangled = demangle(timer_name);
    if (demangled != NULL) {
      eventData = new TimerEventData(START_EVENT, thread_instance::getID(), *demangled);
      delete(demangled);
    } else {
      eventData = new TimerEventData(START_EVENT, thread_instance::getID(), timer_name);
    }
    */
    event_data_ = new timer_event_data(START_EVENT, thread_instance::get_id(), timer_name);
    instance->notify_listeners(event_data_);
}

void stop(string timer_name)
{
    APEX_TIMER_TRACER("stop", timer_name)
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_STOP(timer_name.c_str());
    event_data* event_data_ = NULL;
    // don't do this now
    /*
    string* demangled = demangle(timer_name);
    if (demangled != NULL) {
      eventData = new TimerEventData(STOP_EVENT, thread_instance::getID(), *demangled);
      delete(demangled);
    } else {
      eventData = new TimerEventData(STOP_EVENT, thread_instance::getID(), timer_name);
    }
    */
    event_data_ = new timer_event_data(STOP_EVENT, thread_instance::get_id(), timer_name);
    instance->notify_listeners(event_data_);
}

void stop()
{
    APEX_TIMER_TRACER("stop", "?")
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
    string empty = "";
    event_data* event_data_ = new timer_event_data(STOP_EVENT, thread_instance::get_id(), empty);
    instance->notify_listeners(event_data_);
}

void sample_value(string name, double value)
{
    APEX_TRACER
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    // parse the counter name
    // either /threadqueue{locality#0/total}/length
    // or     /threadqueue{locality#0/worker-thread#0}/length
    sample_value_event_data* event_data_ = NULL;
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
                event_data_ = new sample_value_event_data(tid, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, tid);
            }
            else
            {
                event_data_ = new sample_value_event_data(0, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
            }
        }
        else
        {
            event_data_ = new sample_value_event_data(0, name, value);
            //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
            //TAU_TRIGGER_CONTEXT_EVENT((char *)(name.c_str()), value);
        }
    }
    else
    {
        // what if it doesn't?
        event_data_ = new sample_value_event_data(0, name, value);
        //TAU_TRIGGER_CONTEXT_EVENT((char *)(name.c_str()), value);
    }
    instance->notify_listeners(event_data_);
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
    // exit ALL threads
    //Tau_profile_exit_all_threads();
    //TAU_PROFILE_EXIT("APEX exiting");
    if (!_finalized)
    {
        _finalized = true;
        stringstream ss;
        ss << instance->get_node_id();
        shutdown_event_data* event_data_ = new shutdown_event_data(instance->get_node_id(), thread_instance::get_id());
        instance->notify_listeners(event_data_);
        _notify_listeners = false;
    }
    instance->~apex();
}

void register_thread(string name)
{
    APEX_TRACER
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    //TAU_REGISTER_THREAD();
    // int nid, tid;
    // nid = TAU_PROFILE_GET_NODE();
    // tid = TAU_PROFILE_GET_THREAD();
    //cout << "Node " << nid << " registered thread " << tid << endl;
    // we can't start, because there is no way to stop!
    thread_instance::set_name(name);
    new_thread_event_data* event_data_ = new new_thread_event_data(name);
    instance->notify_listeners(event_data_);
    string::size_type index = name.find("#");
    if (index!=std::string::npos)
    {
        string short_name = name.substr(0,index);
        //cout << "shortening " << name << " to " << shortName << endl;
        start(short_name);
    }
    else
    {
        start(name);
    }
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

    void apex_init(int argc, char** argv)
    {
        APEX_TRACER
        init(argc, argv);
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

    void apex_start(const char * timer_name)
    {
        APEX_TRACER
        start(string(timer_name));
    }

    void apex_stop(const char * timer_name)
    {
        APEX_TRACER
        stop(string(timer_name));
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
        register_thread(string(name));
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

