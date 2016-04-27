//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

// apex main class

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
#include "apex_config.h"
#include "handler.hpp"
#include "event_listener.hpp"
#include "policy_handler.hpp"
#include "profiler_listener.hpp"
#include "apex_options.hpp"
#include "apex_export.h" 
#include "proc_read.h" 
#include <unordered_map>
#if __cplusplus > 201701L 
#include <shared_mutex>
#elif __cplusplus > 201402L
#include <shared_lock>
#else
#include <mutex>
#endif

#ifdef APEX_HAVE_RCR
#include "libenergy.h"
#endif

#if APEX_HAVE_PROC
#include "proc_read.h" 
#endif

#if APEX_HAVE_SOS
#include "sos_handler.hpp" 
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
    apex() : m_argc(0), m_argv(NULL), m_node_id(0), m_my_locality(std::string("0"))
    {
        _initialize();
    };
    apex(int argc, char**argv) : m_argc(argc), m_argv(argv), m_node_id(0), m_my_locality(std::string("0"))
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
#ifdef APEX_HAVE_SOS
    sos_handler * the_sos_handler;
#endif
    profiler_listener * the_profiler_listener;
    proc_data_reader * pd_reader;
    std::string version_string;
    std::vector<event_listener*> listeners;
    std::vector<int (*)()> finalize_functions;
    std::string m_my_locality;
    std::unordered_map<int, std::string> custom_event_names;
#if __cplusplus > 201701L 
    std::shared_mutex custom_event_mutex;
#elif __cplusplus > 201402L
    std::shared_lock custom_event_mutex;
#else
    std::mutex custom_event_mutex;
#endif
    static apex* instance(); // singleton instance
    static apex* instance(int argc, char** argv); // singleton instance
    static apex* __instance(); // special case - for cleanup only!
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
    apex_thread_state get_state(int thread_id) { 
        if ((unsigned int)thread_id >= thread_states.size()) {
            return APEX_IDLE;
        }
        return thread_states[thread_id]; 
    }
    void resize_state(int thread_id) { 
        static std::mutex _mtx;
        if ((unsigned int)thread_id >= thread_states.size()) {
          _mtx.lock();
          if ((unsigned int)thread_id >= thread_states.size()) {
            thread_states.resize(thread_states.size() + 1024); 
          }
          _mtx.unlock();
        }
        thread_states[thread_id] = APEX_IDLE;
    }
    ~apex();
};

int initialize_worker_thread_for_TAU(void);
void init_plugins(void);
void finalize_plugins(void);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

} //namespace apex

