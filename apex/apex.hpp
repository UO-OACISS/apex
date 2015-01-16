//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_HPP
#define APEX_HPP

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

//using namespace std;

namespace apex
{

///////////////////////////////////////////////////////////////////////
// Main class for the APEX project
class apex
{
private:
// private constructors cannot be called
    apex() : m_argc(0), m_argv(NULL), m_node_id(0)
    {
        _initialize();
    };
    apex(int argc, char**argv) : m_argc(argc), m_argv(argv)
    {
        _initialize();
    };
    apex(apex const&) {};            // copy constructor is private
    apex& operator=(apex const& a)
    {
        return const_cast<apex&>(a);
    };  // assignment operator is private
// member variables
    static apex* m_pInstance;
    int m_argc;
    char** m_argv;
    int m_node_id;
    bool m_profiling;
    void _initialize();
    policy_handler * m_policy_handler;
    std::map<int, policy_handler*> period_handlers;
#ifdef APEX_HAVE_HPX3
    hpx::runtime * m_hpx_runtime;
#endif
public:
    std::vector<event_listener*> listeners;
    string* m_my_locality;
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
    ~apex();
};

// These are all static functions for the class. There should be only
// one APEX object in the process space.
APEX_EXPORT void init(const char * thread_name);
APEX_EXPORT void init(int argc, char** argv, const char * thread_name);
APEX_EXPORT void finalize(void);
APEX_EXPORT double version(void);
APEX_EXPORT profiler* start(std::string timer_name);
APEX_EXPORT profiler* start(void * function_address);
APEX_EXPORT void stop(void * profiler);
APEX_EXPORT void resume(void * profiler);
APEX_EXPORT void sample_value(std::string name, double value);
APEX_EXPORT void set_node_id(int id);
APEX_EXPORT void register_thread(std::string name);
APEX_EXPORT void track_power(void);
APEX_EXPORT void track_power_here(void);
APEX_EXPORT void enable_tracking_power(void);
APEX_EXPORT void disable_tracking_power(void);
APEX_EXPORT void set_interrupt_interval(int seconds);
// Method for registering event-based policy
APEX_EXPORT apex_policy_handle* register_policy(const apex_event_type when, std::function<bool(apex_context const&)> f);
// Method for registering periodic policy
APEX_EXPORT apex_policy_handle* register_periodic_policy(unsigned long period, std::function<bool(apex_context const&)> f);
APEX_EXPORT apex_profile* get_profile(apex_function_address function_address);
APEX_EXPORT apex_profile* get_profile(string &timer_name);
APEX_EXPORT std::vector<std::string> get_available_profiles();

#ifdef APEX_HAVE_HPX3
hpx::runtime * get_hpx_runtime_ptr(void);
#endif
} //namespace apex

#endif //APEX_HPP
