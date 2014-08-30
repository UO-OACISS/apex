//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_HPP
#define APEX_HPP

#include <string>
#include <vector>
#include <stdint.h>
#include "apex_types.h"
#include "handler.hpp"
#include "event_listener.hpp"
#include "policy_handler.hpp"
//#include <chrono.h>

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
    std::vector<event_listener*> listeners;
public:
    string* m_my_locality;
    static apex* instance(); // singleton instance
    static apex* instance(int argc, char** argv); // singleton instance
    void set_node_id(int id);
    int get_node_id(void);
    void notify_listeners(event_data* event_data_);
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
void init(void);
void init(int argc, char** argv);
void finalize(void);
double version(void);
void start(std::string timer_name);
void stop(std::string timer_name);
void start(void * function_address);
void stop(void * function_address);
void stop(void);
void sample_value(std::string name, double value);
void set_node_id(int id);
void register_thread(std::string name);
void track_power(void);
void track_power_here(void);
void enable_tracking_power(void);
void disable_tracking_power(void);
void set_interrupt_interval(int seconds);
// Method for registering event-based policy
apex_policy_handle* register_policy(const apex_event_type when,
                    std::function<bool(apex_context const&)> f);
// Method for registering periodic policy
/*
template <typename Rep, typename Period>
apex_policy_handle register_policy(
    std::chrono::duration<Rep, Period> const& period,
    std::function<bool(apex_context const&)> f);
*/

apex_policy_handle* register_periodic_policy(unsigned long period, std::function<bool(apex_context const&)> f);
}

#endif //APEX_HPP
