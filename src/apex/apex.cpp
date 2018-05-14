//  Copyright (c) 2014 University of Oregon
//

#ifdef APEX_HAVE_HPX
#include <hpx/config.hpp>
#endif

#include "apex.hpp"
#include "apex_api.hpp"
#include "apex_types.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <utility>
#include <memory>
#include <algorithm>
#if APEX_USE_PLUGINS
#include <dlfcn.h>
#endif
//#include <cxxabi.h> // this is for demangling strings.

#include "concurrency_handler.hpp"
#include "policy_handler.hpp"
#include "thread_instance.hpp"
#include "utils.hpp"

#include "tau_listener.hpp"
#include "profiler_listener.hpp"
#if defined(APEX_DEBUG) || defined(APEX_ERROR_HANDLING)
// #define APEX_DEBUG_disabled
#include "apex_error_handling.hpp"
#endif
#include "address_resolution.hpp"

#ifdef APEX_HAVE_OTF2
#include "otf2_listener.hpp"
#endif

#ifdef APEX_HAVE_RCR
#include "libenergy.h"
#endif

#if APEX_HAVE_PROC
#include "proc_read.h"
#endif

#ifdef APEX_HAVE_HPX
#include <boost/assign.hpp>
#include <boost/cstdint.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos/local/composable_guard.hpp>
static void apex_schedule_shutdown(void);
#endif

#if 0
#define FUNCTION_ENTER printf("enter %lu *** %s!\n", thread_instance::get_id(), __func__); fflush(stdout);
#define FUNCTION_EXIT  printf("exit  %lu *** %s!\n", thread_instance::get_id(), __func__); fflush(stdout);
#else
#define FUNCTION_ENTER
#define FUNCTION_EXIT
#endif

APEX_NATIVE_TLS bool _registered = false;
APEX_NATIVE_TLS bool _exited = false;
static bool _initialized = false;

using namespace std;

namespace apex
{

// Global static pointer used to ensure a single instance of the class.
std::atomic<apex*> apex::m_pInstance(nullptr);

std::atomic<bool> _notify_listeners(true);
std::atomic<bool> _measurement_stopped(false);
APEX_NATIVE_TLS profiler * top_level_timer = nullptr;

/*
 * The destructor will request power data from RCRToolkit
 */
apex::~apex()
{
#ifdef APEX_HAVE_RCR
    //cout << "Getting energy..." << endl;
    energyDaemonTerm();
#endif
    for (unsigned int i = listeners.size(); i > 0 ; i--) {
        event_listener * el = listeners[i-1];
        listeners.pop_back();
        delete el;
    }
#if APEX_HAVE_PROC
    if (pd_reader != nullptr) {
        delete pd_reader;
    }
#endif
    m_pInstance = nullptr;
    while (apex_policy_handles.size() > 0) {
        auto tmp = apex_policy_handles.back();
        apex_policy_handles.pop_back();
        delete(tmp);
    }
}

int apex::get_node_id()
{
    return m_node_id;
}

int apex::get_num_ranks()
{
    return m_num_ranks;
}

#ifdef APEX_HAVE_HPX
static void init_hpx_runtime_ptr(void) {
    if (apex_options::disable() == true) { return; }
    apex * instance = apex::instance();
    if(instance != nullptr) {
        hpx::runtime * runtime = hpx::get_runtime_ptr();
        instance->set_hpx_runtime(runtime);
/*
        std::stringstream ss;
        ss << "/threads{locality#" << instance->get_node_id() << "/total}/count/cumulative";
        instance->setup_runtime_counter(ss.str());
*/
    }
}

static void finalize_hpx_runtime(void) {
    FUNCTION_ENTER
    if (apex_options::disable() == true) { return; }
    static std::mutex init_mutex;
    static bool hpx_finalized = false;
    unique_lock<mutex> l(init_mutex);
    if (hpx_finalized) { return; }
    apex * instance = apex::instance();
    // Get the HPX counters one (last?) time, at exit
    if(instance != nullptr) {
        if(hpx::get_runtime_ptr() != nullptr) {
            instance->query_runtime_counters();
        }
    }
    // Tell other localities to shutdown APEX
    //apex_schedule_shutdown();
    // Shutdown APEX
    //finalize();
    hpx_finalized = true;
    FUNCTION_EXIT
}
#endif

/*
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void apex::_initialize()
{
#if defined(APEX_DEBUG) || defined(APEX_ERROR_HANDLING)
    //apex_register_signal_handler();
#endif
    this->m_pInstance = this;
    this->m_policy_handler = nullptr;
    stringstream ss;
    ss << "locality#" << this->m_node_id;
    this->m_my_locality = string(ss.str());
    stringstream tmp;
#if defined (GIT_TAG)
    tmp << GIT_TAG;
#else
    tmp << APEX_VERSION_MAJOR + (APEX_VERSION_MINOR/10.0);
#endif
#if defined (GIT_COMMIT_HASH)
    tmp << "-" << GIT_COMMIT_HASH ;
#endif
#if defined (GIT_BRANCH)
    tmp << "-" << GIT_BRANCH ;
#endif
    tmp << std::endl << "Built on: " << __TIME__ << " " << __DATE__;
    tmp << std::endl << "C++ Language Standard version : " << __cplusplus;
#if defined(__clang__)
    /* Clang/LLVM. ---------------------------------------------- */
    tmp << std::endl << "Clang Compiler version : " << __VERSION__;
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC. ------------------------------------------ */
    tmp << std::endl << "Intel Compiler version : " << __VERSION__;
#elif defined(__GNUC__) || defined(__GNUG__)
    /* GNU GCC/G++. --------------------------------------------- */
    tmp << std::endl << "GCC Compiler version : " << __VERSION__;
#elif defined(__HP_cc) || defined(__HP_aCC)
    /* Hewlett-Packard C/aC++. ---------------------------------- */
    tmp << std::endl << "HP Compiler version : " << __HP_aCC;
#elif defined(__IBMC__) || defined(__IBMCPP__)
    /* IBM XL C/C++. -------------------------------------------- */
    tmp << std::endl << "IBM Compiler version : " << __xlC__;
#elif defined(_MSC_VER)
    /* Microsoft Visual Studio. --------------------------------- */
    tmp << std::endl << "Microsoft Compiler version : " << _MSC_FULL_VER;
#elif defined(__PGI)
    /* Portland Group PGCC/PGCPP. ------------------------------- */
    tmp << std::endl << "PGI Compiler version : " << __VERSION__;
#elif defined(__SUNPRO_CC)
    /* Oracle Solaris Studio. ----------------------------------- */
    tmp << std::endl << "Oracle Compiler version : " << __SUNPRO_CC;
#endif

    this->version_string = std::string(tmp.str().c_str());
#ifdef APEX_HAVE_HPX
    this->m_hpx_runtime = nullptr;
    hpx::register_startup_function(init_hpx_runtime_ptr);
    hpx::register_pre_shutdown_function(finalize_hpx_runtime);
#endif
#ifdef APEX_HAVE_RCR
    energyDaemonInit();
#endif
#ifdef APEX_HAVE_MSR
    apex_init_msr();
#endif
    bool tau_loaded = false;
    if (apex_options::use_tau())
    {
        // before spawning any other threads, initialize TAU.
        char * tmp = const_cast<char*>("APEX");
        char * argv[] = {tmp};
        int argc = 1;
        tau_loaded = tau_listener::initialize_tau(argc, argv);
    }
    {
        //write_lock_type l(listener_mutex);
        this->the_profiler_listener = new profiler_listener();
        // this is always the first listener!
           listeners.push_back(the_profiler_listener);
        if (apex_options::use_tau() && tau_loaded)
        {
            listeners.push_back(new tau_listener());
        }
#ifdef APEX_HAVE_OTF2
        if (apex_options::use_otf2())
        {
            listeners.push_back(new otf2_listener());
        }
#endif

/* For the Jupyter support, always enable the concurrency handler. */
#ifndef APEX_WITH_JUPYTER_SUPPORT
        if (apex_options::use_concurrency() > 0)
#endif
        {
            listeners.push_back(new concurrency_handler(apex_options::concurrency_period(), apex_options::use_concurrency()));
        }
        startup_throttling();
/* For the Jupyter support, always enable the policy listener. */
#ifndef APEX_WITH_JUPYTER_SUPPORT
        if (apex_options::use_policy())
#endif
        {
            this->m_policy_handler = new policy_handler();
            listeners.push_back(this->m_policy_handler);
        }
    }
#if APEX_HAVE_PROC
    if (apex_options::use_proc_cpuinfo() ||
        apex_options::use_proc_meminfo() ||
        apex_options::use_proc_net_dev() ||
        apex_options::use_proc_self_status() ||
        apex_options::use_proc_stat()) {
        pd_reader = new proc_data_reader();
    } else {
        pd_reader = nullptr;
    }
#endif
    this->resize_state(1);
    this->set_state(0, APEX_BUSY);
}

apex* apex::instance()
{
    static std::mutex init_mutex;
    // Only allow one instance of class to be generated.
    if (m_pInstance == nullptr) {
        if (_measurement_stopped) {
            return nullptr;
        } else {
            unique_lock<mutex> l(init_mutex);
            if (m_pInstance == nullptr) {
                m_pInstance = new apex();
            }
        }
    }
    return m_pInstance;
}

/* This function is used to set up thread-specific data structures
 * for each of the asynchronous threads in APEX. For example, the
 * proc_read thread needs a queue for processing sampled values.
 */
void apex::async_thread_setup() {
    apex* instance = apex::instance();
    instance->the_profiler_listener->async_thread_setup();
}

// special case - for cleanup only!
apex* apex::__instance()
{
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
        //write_lock_type l(listener_mutex);
        listeners.push_back(period_handlers[period]);
    }
    return period_handlers[period];
}

#ifdef APEX_HAVE_HPX
void apex::set_hpx_runtime(hpx::runtime * hpx_runtime) {
    m_hpx_runtime = hpx_runtime;
}

hpx::runtime * apex::get_hpx_runtime(void) {
    return m_hpx_runtime;
}
#endif

uint64_t init(const char * thread_name, uint64_t comm_rank, uint64_t comm_size) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return APEX_ERROR; }
    // protect against multiple initializations
#ifdef APEX_WITH_JUPYTER_SUPPORT
    if (_registered || _initialized) { 
        // reset all counters, and return.
        reset(APEX_NULL_FUNCTION_ADDRESS);
        return APEX_NOERROR; 
    }
#else
    if (_registered || _initialized) { 
        /* check to see if APEX was initialized by OMPT before MPI had a chance
         * to pass in any values */
        if ((comm_rank < comm_size) && (comm_size > 1)) { // simple validation
            apex* instance = apex::instance(); // get/create the Apex static instance
            instance->set_node_id(comm_rank);
            instance->set_num_ranks(comm_size);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->set_node_id((int)comm_rank, (int)comm_size);
            }
        }
        return APEX_ERROR; 
    }
#endif
    _registered = true;
    _initialized = true;
    apex* instance = apex::instance(); // get/create the Apex static instance
    // assign the rank and size.  Why not in the constructor?
    // because, if we registered a startup policy, the default
    // constructor was called, without the correct comm_rank and comm_size.
    if (comm_rank < comm_size && comm_size > 0) { // simple validation
      instance->set_node_id(comm_rank);
      instance->set_num_ranks(comm_size);
    }
    //printf("Node %lu of %lu\n", comm_rank, comm_size);
    if (!instance || _exited) return APEX_ERROR; // protect against calls after finalization
    init_plugins();
    startup_event_data data(comm_rank, comm_size);
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(data);
        }
    }
    if (apex_options::top_level_os_threads()) {
        // start top-level timers for threads
        if (thread_name) {
            stringstream ss;
            ss << "OS Thread: " << thread_name;
            top_level_timer = start(ss.str().c_str());
        } else {
            top_level_timer = start("OS Thread");
        }
    }
    if (apex_options::use_screen_output() && instance->get_node_id() == 0) {
      std::cout << version() << std::endl;
      apex_options::print_options();
    }
    if (apex_options::throttle_energy() && apex_options::throttle_concurrency() ) {
      setup_power_cap_throttling();
    }
    // this code should be absorbed from "new node" event to "on_startup" event.
    node_event_data node_data(comm_rank, thread_instance::get_id());
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_node(node_data);
        }
    }
    /* register the finalization function, for program exit */
    std::atexit(cleanup);
    return APEX_NOERROR;
}

string& version() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { static string tmp("disabled"); return tmp; }
    apex* instance = apex::instance(); // get the Apex static instance
    return instance->version_string;
}

/* Populate the new task_wrapper object, and notify listeners. */
inline std::shared_ptr<task_wrapper> _new_task(
    task_identifier * id, 
    const uint64_t task_id,
    const std::shared_ptr<task_wrapper> &parent_task, apex* instance) {
    std::shared_ptr<task_wrapper> tt_ptr = make_shared<task_wrapper>();
    tt_ptr->task_id = id;
    // was a parent passed in?
    if (parent_task != nullptr) {
        tt_ptr->parent_guid = parent_task->guid;
        tt_ptr->parent = parent_task;
    } else {
        // if not, is there a current timer?
        profiler * p = thread_instance::instance().get_current_profiler();
        if (p != nullptr) {
            tt_ptr->parent_guid = p->guid;
            tt_ptr->parent = p->tt_ptr;
        } else {
            tt_ptr->parent = task_wrapper::get_apex_main_wrapper();
            // tt_ptr->parent_guid is 0 by default
        }
    }
    if (task_id == UINTMAX_MAX) {
        // generate a GUID
        tt_ptr->guid = thread_instance::get_guid();
    } else {
        // use the runtime provided GUID
        tt_ptr->guid = task_id;
    }
    //instance->active_task_wrappers.insert(tt_ptr);
    return tt_ptr;
}

profiler* start(const std::string &timer_name)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_START
        return nullptr; 
    }
    //printf("%lu: %s\n", thread_instance::get_id(), timer_name.c_str()); fflush(stdout);
    static const std::string apex_internal("apex_internal");
    if (starts_with(timer_name, apex_internal)) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
        return profiler::get_disabled_profiler(); // don't process our own events - queue scrubbing tasks.
    }
#ifdef APEX_HAVE_HPX_disabled
    // Finalize at the _start_ of HPX shutdown so that we can stop any
    // outstanding hpx::util::interval_timer instances. If any are left
    // running, HPX shutdown will never complete.
    //if (starts_with(timer_name, string("shutdown_all"))) {
    static const std::string shutdown_str("shutdown_all_action");
    if (timer_name.compare(shutdown_str) == 0) {
        APEX_UTIL_REF_COUNT_HPX_SHUTDOWN_START
        finalize();
        return profiler::get_disabled_profiler();
    }
    static const std::string periodic_str("at_timer (expire at)");
    if (timer_name.compare(periodic_str) == 0) {
        APEX_UTIL_REF_COUNT_HPX_TIMER_START
        return profiler::get_disabled_profiler();
    }
#endif
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE
        return nullptr;
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_START
        return profiler::get_disabled_profiler(); 
    }
    std::shared_ptr<task_wrapper> tt_ptr(nullptr);
    if (_notify_listeners) {
        bool success = true;
        task_identifier * id = task_identifier::get_task_id(timer_name);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        //read_lock_type l(instance->listener_mutex);
        //cout << thread_instance::get_id() << " Start : " << id->get_name() << endl; fflush(stdout);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " << id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                return profiler::get_disabled_profiler();
            }
        }
    }
    static std::string apex_process_profile_str("apex::process_profiles");
    if (timer_name.compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
    } else {
        APEX_UTIL_REF_COUNT_START
    }
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* start(const apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_START
        return nullptr; 
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE
        return nullptr;
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_START
        return profiler::get_disabled_profiler(); 
    }
    std::shared_ptr<task_wrapper> tt_ptr(nullptr);
    if (_notify_listeners) {
        bool success = true;
        task_identifier * id = task_identifier::get_task_id(function_address);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        //cout << thread_instance::get_id() << " Start : " << id->get_name() << endl; fflush(stdout);
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " << id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                return profiler::get_disabled_profiler();
            }
        }
    }
    APEX_UTIL_REF_COUNT_START
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

void debug_print(const char * event, std::shared_ptr<task_wrapper> &tt_ptr) {
    static std::mutex this_mutex;
    std::unique_lock<std::mutex> l(this_mutex);
    if (tt_ptr == nullptr) {
        cout << thread_instance::get_id() << " " << event << " : (null) : (null)" 
            << endl; fflush(stdout);
    } else {
        cout << thread_instance::get_id() << " " << event << " : " << tt_ptr->guid << " : " <<
            tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
    }
}

profiler* start(std::shared_ptr<task_wrapper> &tt_ptr) {
#if defined(APEX_DEBUG_disabled)
    debug_print("Start", tt_ptr);
#endif
    if (tt_ptr == nullptr) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
        return nullptr;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_START
        return nullptr; 
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE
        return nullptr;
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_START
        return profiler::get_disabled_profiler(); 
    }
    if (_notify_listeners) {
        bool success = true;
        //cout << thread_instance::get_id() << " Start : " << id->get_name() << endl; fflush(stdout);
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            tt_ptr->prof = thread_instance::instance().get_current_profiler();
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " << id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                return profiler::get_disabled_profiler();
            }
        }
    }
    APEX_UTIL_REF_COUNT_START
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* resume(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_RESUME
        return nullptr; 
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_RESUME
        return profiler::get_disabled_profiler(); 
    }
    // don't process our own events
    if (starts_with(timer_name, string("apex_internal"))) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
        return profiler::get_disabled_profiler();
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_RESUME_AFTER_FINALIZE
        return nullptr;
    }
    std::shared_ptr<task_wrapper> tt_ptr(nullptr);
    if (_notify_listeners) {
        task_identifier * id = task_identifier::get_task_id(timer_name);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        try {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_resume(tt_ptr);
            }
        } catch (disabled_profiler_exception e) { 
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler(); 
        }
    }
    static std::string apex_process_profile_str("apex::process_profiles");
    if (timer_name.compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
    } else {
        APEX_UTIL_REF_COUNT_RESUME
    }
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* resume(const apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_RESUME
        return nullptr; 
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_RESUME
        return profiler::get_disabled_profiler(); 
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_RESUME_AFTER_FINALIZE
        return nullptr;
    }
    std::shared_ptr<task_wrapper> tt_ptr(nullptr);
    if (_notify_listeners) {
        task_identifier * id = task_identifier::get_task_id(function_address);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        try {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_resume(tt_ptr);
            }
        } catch (disabled_profiler_exception e) { 
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler(); 
        }
    }
    APEX_UTIL_REF_COUNT_RESUME
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* resume(profiler * p) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_RESUME
        return nullptr; 
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { 
        APEX_UTIL_REF_COUNT_SUSPENDED_RESUME
        return profiler::get_disabled_profiler(); 
    }
    if (p->stopped) {
        APEX_UTIL_REF_COUNT_DOUBLE_STOP
        return profiler::get_disabled_profiler(); 
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_RESUME_AFTER_FINALIZE
        return nullptr;
    }
    p->restart();
    if (_notify_listeners) {
        try {
            // skip the profiler_listener - we are restoring a child timer
            // for a parent that was yielded.
            for (unsigned int i = 1 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_resume(p->tt_ptr);
            }
        } catch (disabled_profiler_exception e) { 
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler(); 
        }
    }
    static std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
    } else {
        APEX_UTIL_REF_COUNT_RESUME
    }
    return p;
}

void reset(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    task_identifier * id = task_identifier::get_task_id(timer_name);
    //instance->the_profiler_listener->reset(id);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_reset(id);
        }
    }
}

void reset(apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    task_identifier * id = nullptr;
    if (function_address != APEX_NULL_FUNCTION_ADDRESS) {
        id = task_identifier::get_task_id(function_address);
    }
    //instance->the_profiler_listener->reset(id);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_reset(id);
        }
    }
}

void set_state(apex_thread_state state) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    instance->set_state(thread_instance::get_id(), state);
}

void stop(profiler* the_profiler, bool cleanup) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_STOP
        return; 
    }
    if (the_profiler == profiler::get_disabled_profiler()) {
        APEX_UTIL_REF_COUNT_DISABLED_STOP
        return; // profiler was throttled.
    }
    if (the_profiler == nullptr) {
        APEX_UTIL_REF_COUNT_NULL_STOP
        return;
    }
    if (the_profiler->stopped) {
        APEX_UTIL_REF_COUNT_DOUBLE_STOP
        return;
    }
    thread_instance::instance().clear_current_profiler(the_profiler, false, null_task_wrapper);
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited || _measurement_stopped) { 
        APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE
        return; 
    }
    std::shared_ptr<profiler> p{the_profiler};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(p);
        }
    }
    //cout << thread_instance::get_id() << " Stop : " << the_profiler->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
    static std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
    } else {
        APEX_UTIL_REF_COUNT_STOP
    }
    if (cleanup) {
        if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_task_complete(p->tt_ptr);
            }
        }
        //instance->active_task_wrappers.erase(p->tt_ptr);
        p->tt_ptr = nullptr;
    }
}

void stop(std::shared_ptr<task_wrapper> &tt_ptr) {
#if defined(APEX_DEBUG_disabled)
    debug_print("Stop", tt_ptr);
#endif
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_STOP
        tt_ptr = nullptr;
        return; 
    }
    if (tt_ptr == nullptr || tt_ptr->prof == nullptr) {
        APEX_UTIL_REF_COUNT_NULL_STOP
        return;
    }
    apex* instance = apex::instance(); // get the Apex static instance
    if (tt_ptr->prof == profiler::get_disabled_profiler()) {
        APEX_UTIL_REF_COUNT_DISABLED_STOP
        return; // profiler was throttled.
    }
    if (tt_ptr->prof->stopped) {
        APEX_UTIL_REF_COUNT_DOUBLE_STOP
        return;
    }
    thread_instance::instance().clear_current_profiler(tt_ptr->prof, false, null_task_wrapper);
    // protect against calls after finalization
    if (!instance || _exited || _measurement_stopped) { 
        APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE
        return; 
    }
    std::shared_ptr<profiler> p{tt_ptr->prof};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(p);
        }
    }
    //cout << thread_instance::get_id() << " Stop : " << tt_ptr->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
    static std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
    } else {
        APEX_UTIL_REF_COUNT_STOP
    }
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_task_complete(tt_ptr);
        }
    }
}

void yield(profiler* the_profiler)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_YIELD
        return; 
    }
    if (the_profiler == profiler::get_disabled_profiler()) { 
        APEX_UTIL_REF_COUNT_DISABLED_YIELD
        return; // profiler was throttled.
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited || _measurement_stopped) {
        APEX_UTIL_REF_COUNT_YIELD_AFTER_FINALIZE
        return;
    }
    if (the_profiler == nullptr) {
        APEX_UTIL_REF_COUNT_NULL_YIELD
        return;
    }
    if (the_profiler->stopped) {
        APEX_UTIL_REF_COUNT_DOUBLE_YIELD
        return;
    }
    thread_instance::instance().clear_current_profiler(the_profiler, false, null_task_wrapper);
    std::shared_ptr<profiler> p{the_profiler};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_yield(p);
        }
    }
    //cout << thread_instance::get_id() << " Yield : " << the_profiler->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
    static std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD
    } else {
        APEX_UTIL_REF_COUNT_YIELD
    }
}

void yield(std::shared_ptr<task_wrapper> &tt_ptr)
{
#if defined(APEX_DEBUG_disabled)
    debug_print("Yield", tt_ptr);
#endif
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { 
        APEX_UTIL_REF_COUNT_DISABLED_YIELD
        return; 
    }
    if (tt_ptr == nullptr || tt_ptr->prof == nullptr) {
        APEX_UTIL_REF_COUNT_NULL_YIELD
        return;
    }
    if (tt_ptr->prof == profiler::get_disabled_profiler()) { 
        APEX_UTIL_REF_COUNT_DISABLED_YIELD
        return; // profiler was throttled.
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited || _measurement_stopped) {
        APEX_UTIL_REF_COUNT_YIELD_AFTER_FINALIZE
        return;
    }
    if (tt_ptr->prof->stopped) {
        APEX_UTIL_REF_COUNT_DOUBLE_YIELD
        return;
    }
    thread_instance::instance().clear_current_profiler(tt_ptr->prof, true, tt_ptr);
    std::shared_ptr<profiler> p{tt_ptr->prof};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_yield(p);
        }
    }
    //cout << thread_instance::get_id() << " Yield : " << tt_ptr->prof->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
    static std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD
    } else {
        APEX_UTIL_REF_COUNT_YIELD
    }
    tt_ptr->prof = nullptr;
}

void sample_value(const std::string &name, double value)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    // parse the counter name
    // either /threadqueue{locality#0/total}/length
    // or     /threadqueue{locality#0/worker-thread#0}/length
    int tid = 0;
    if (name.find(instance->m_my_locality) != name.npos)
    {
        if (name.find("worker-thread") != name.npos)
        {
            string tmp_name = string(name.c_str());
            // tokenize by / character
            char* token = strtok(const_cast<char*>(tmp_name.c_str()), "/");
            while (token!=nullptr) {
              if (strstr(token, "worker-thread")==NULL) { break; }
              token = strtok(NULL, "/");
            }
            if (token != nullptr) {
              // strip the trailing close bracket
              token = strtok(token, "}");
              tid = thread_instance::map_name_to_id(token);
            }
            if (tid == -1) {
                tid = 0;
            }
        }
    }
    sample_value_event_data data(tid, name, value);
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_sample_value(data);
        }
    }
}

std::shared_ptr<task_wrapper> new_task(
    const std::string &timer_name, 
    const uint64_t task_id, 
    const std::shared_ptr<task_wrapper> &parent_task)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return nullptr; }
    static const std::string apex_internal("apex_internal");
    if (starts_with(timer_name, apex_internal)) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; // don't process our own events - queue scrubbing tasks.
    }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) { 
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; 
    } // protect against calls after finalization
    task_identifier * id = task_identifier::get_task_id(timer_name);
    std::shared_ptr<task_wrapper> tt_ptr(_new_task(id, task_id, parent_task, instance));
    APEX_UTIL_REF_COUNT_TASK_WRAPPER
    return tt_ptr;
}

std::shared_ptr<task_wrapper> new_task(
    const apex_function_address function_address, 
    const uint64_t task_id, 
    const std::shared_ptr<task_wrapper> &parent_task) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return nullptr; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) { return nullptr; } // protect against calls after finalization
    task_identifier * id = task_identifier::get_task_id(function_address);
    std::shared_ptr<task_wrapper> tt_ptr(_new_task(id, task_id, parent_task, instance));
    return tt_ptr;
}

std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> &wrapper, 
    const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return nullptr; }
    assert(wrapper);
    task_identifier * id = task_identifier::get_task_id(timer_name);
    if (id != wrapper->get_task_id()) {
        wrapper->aliases.insert(id);
    }
    return wrapper;
}

std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> &wrapper, 
    const apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return nullptr; }
    if (wrapper == nullptr) {
        apex* instance = apex::instance(); // get the Apex static instance
        if (!instance || _exited) { return nullptr; } // protect against calls after finalization
        task_identifier * id = task_identifier::get_task_id(function_address);
        wrapper = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
    } else {
        wrapper->task_id = task_identifier::get_task_id(function_address);
    }
    return wrapper;
}

std::atomic<int> custom_event_count(APEX_CUSTOM_EVENT_1);

apex_event_type register_custom_event(const std::string &name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return APEX_CUSTOM_EVENT_1; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return APEX_CUSTOM_EVENT_1; // protect against calls after finalization
    if (custom_event_count == APEX_MAX_EVENTS) {
      std::cerr << "Cannot register more than MAX Events! (set to " << APEX_MAX_EVENTS << ")" << std::endl;
    }
    write_lock_type l(instance->custom_event_mutex);
    instance->custom_event_names[custom_event_count] = name;
    int tmp = custom_event_count;
    custom_event_count++;
    return (apex_event_type)tmp;
}

void custom_event(apex_event_type event_type, void * custom_data) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    custom_event_data data(event_type, custom_data);
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_custom_event(data);
        }
    }
}

#ifdef APEX_HAVE_HPX
hpx::runtime * get_hpx_runtime_ptr(void) {
    apex * instance = apex::instance();
    if (!instance || _exited) {
        return nullptr;
    }
    hpx::runtime * runtime = instance->get_hpx_runtime();
    return runtime;
}
#endif

void init_plugins(void) {
    if (apex_options::disable() == true) { return; }
#ifdef APEX_USE_PLUGINS
    std::string plugin_names_str{apex_options::plugins()};
    std::string plugins_prefix{apex_options::plugins_path()};
    std::string plugins_suffix{".so"};
    if(plugin_names_str.empty()) {
        return;
    }
    std::vector<std::string> plugin_names;
    std::vector<std::string> plugin_paths;
    split( plugin_names_str, ':', plugin_names);
    for(const std::string & plugin_name : plugin_names) {
        plugin_paths.push_back(plugins_prefix + "/" + plugin_name + plugins_suffix);
    }
    for(const std::string & plugin_path : plugin_paths) {
        const char * path = plugin_path.c_str();
        void * plugin_handle = dlopen(path, RTLD_NOW);
        if(!plugin_handle) {
            std::cerr << "Error loading plugin " << path << ": " << dlerror() << std::endl;
            continue;
        }
        int (*init_fn)() = (int (*)()) ((uintptr_t) dlsym(plugin_handle, "apex_plugin_init"));
        if(!init_fn) {
            std::cerr << "Error loading apex_plugin_init from " << path << ": " << dlerror() << std::endl;
            dlclose(plugin_handle);
            continue;
        }
        int (*finalize_fn)() = (int (*)()) ((uintptr_t) dlsym(plugin_handle, "apex_plugin_finalize"));
        if(!finalize_fn) {
            std::cerr << "Error loading apex_plugin_finalize from " << path << ": " << dlerror() << std::endl;
            dlclose(plugin_handle);
            continue;
        }
        apex * instance = apex::instance();
        if(!instance) {
            std::cerr << "Error getting APEX instance while registering finalize function from " << path << std::endl;
            continue;
        }
        instance->finalize_functions.push_back(finalize_fn);
        int result = init_fn();
        if(result != 0) {
            std::cerr << "Error: apex_plugin_init for " << path << " returned " << result << std::endl;
            dlclose(plugin_handle);
            continue;
        }
    }
#endif
}

void finalize_plugins(void) {
    FUNCTION_ENTER
    if (apex_options::disable() == true) { return; }
#ifdef APEX_USE_PLUGINS
    apex * instance = apex::instance();
    if(!instance) return;
    for(int (*finalize_function)() : instance->finalize_functions) {
        int result = finalize_function();
        if(result != 0) {
            std::cerr << "Error: plugin finalize function returned " << result << std::endl;
            continue;
        }
    }
#endif
    FUNCTION_EXIT
}

std::string dump(bool reset) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return(std::string("")); }
    bool old_screen_output = apex_options::use_screen_output();
#ifdef APEX_WITH_JUPYTER_SUPPORT
    // force output in the Jupyter notebook
    apex_options::use_screen_output(true);
#endif

    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) { FUNCTION_EXIT return(std::string("")); } // protect against calls after finalization
    if (_notify_listeners) {
        dump_event_data data(instance->get_node_id(), thread_instance::get_id(), reset);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_dump(data);
        }
        apex_options::use_screen_output(old_screen_output);
        return(data.output);
    }
    apex_options::use_screen_output(old_screen_output);
    return(std::string(""));
}

void finalize()
{
#ifdef APEX_WITH_JUPYTER_SUPPORT
    // reset all counters, and return.
    //reset(APEX_NULL_FUNCTION_ADDRESS);
    return;
#endif
    FUNCTION_ENTER
    // prevent re-entry, be extra strict about race conditions - it is possible.
    mutex shutdown_mutex;
    static bool finalized = false;
    {
        unique_lock<mutex> l(shutdown_mutex);
        if (finalized) { FUNCTION_EXIT return; };
        finalized = true;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // FIRST, stop the top level timer, while the infrastructure is still functioning.
    if (top_level_timer != nullptr) { stop(top_level_timer); }
    // stop processing new timers/counters/messages/tasks/etc.
    apex_options::suspend(true);
    // now, process all output
    dump(false);
    // if not done already...
    shutdown_throttling();
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) { FUNCTION_EXIT return; } // protect against calls after finalization
    finalize_plugins();
    instance->stop_all_policy_handles();
#if APEX_HAVE_PROC
    if (instance->pd_reader != nullptr) {
        instance->pd_reader->stop_reading();
    }
#endif
    exit_thread();
#if APEX_HAVE_MSR
    apex_finalize_msr();
#endif
    if (!_measurement_stopped)
    {
        _measurement_stopped = true;
        APEX_UTIL_REPORT_STATS
#ifdef APEX_HAVE_HPX
        /* HPX shutdown happens on a new thread. We don't want
         * to register a new thread. */
        shutdown_event_data data(instance->get_node_id(), 0);
#else
        shutdown_event_data data(instance->get_node_id(), thread_instance::get_id());
#endif
        _notify_listeners = false;
        {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_shutdown(data);
            }
        }
    }
    thread_instance::delete_instance();
#if APEX_HAVE_BFD
    address_resolution::delete_instance();
#endif
    FUNCTION_EXIT
}

void cleanup(void) {
    FUNCTION_ENTER
#ifdef APEX_HAVE_HPX
    // prevent crash at shutdown.
    return;
#endif
    // prevent re-entry, be extra strict about race conditions - it is possible.
    mutex shutdown_mutex;
    static bool finalized = false;
    {
        unique_lock<mutex> l(shutdown_mutex);
        if (finalized) { return; };
        finalized = true;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { FUNCTION_EXIT return; }
    apex* instance = apex::__instance(); // get the Apex static instance
    if (!instance) { FUNCTION_EXIT return; } // protect against multiple calls
    if (!_measurement_stopped) {
        finalize();
    }
    /* this is one of the last things we should do - because the apex_options
     * sometimes control behavior at shutdown. */
    apex_options::delete_instance();
    /*
    for (auto t : instance->active_task_wrappers) {
        instance->active_task_wrappers.erase(t);
    }
    */
    delete(instance);
    FUNCTION_EXIT
}

void register_thread(const std::string &name)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    if (_registered) return; // protect against multiple registrations on the same thread
    _registered = true;
    thread_instance::set_name(name);
    instance->resize_state(thread_instance::get_id());
    instance->set_state(thread_instance::get_id(), APEX_BUSY);
    new_thread_event_data data(name);
    if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_thread(data);
        }
    }
    if (apex_options::top_level_os_threads()) {
        stringstream ss;
        ss << "OS Thread: ";
        // start top-level timers for threads
        if (name.find("worker-thread") != name.npos) {
            ss << "worker-thread";
            top_level_timer = start(ss.str());
        } else {
            string::size_type index = name.find("#");
            if (index!=std::string::npos) {
                string short_name = name.substr(0,index);
                ss << short_name;
                top_level_timer = start(ss.str());
            } else {
                ss << name;
                top_level_timer = start(ss.str());
            }
        }
    }
}

void exit_thread(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    if (top_level_timer != nullptr) { stop(top_level_timer); }
    _exited = true;
    event_data data;
    if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_exit_thread(data);
        }
    }
    // delete the thread local instance
    if (thread_instance::get_id() != 0) {
        thread_instance::delete_instance();
    }
}

void apex::push_policy_handle(apex_policy_handle* handle) {
    apex_policy_handles.push_back(handle);
}

void apex::pop_policy_handle(apex_policy_handle* handle) {
    apex_policy_handles.remove(handle);
}

bool apex::policy_handle_exists(apex_policy_handle* handle) {
    return (std::find(apex_policy_handles.begin(),
                      apex_policy_handles.end(), 
                      handle) != apex_policy_handles.end());
}

void apex::stop_all_policy_handles(void) {
    while (apex_policy_handles.size() > 0) {
        auto tmp = apex_policy_handles.back();
        // this will pop, deregister and delete the handle
        deregister_policy(tmp);
    }
}

apex_policy_handle* register_policy(const apex_event_type when,
                    std::function<int(apex_context const&)> f)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    int id = -1;
    policy_handler * handler = apex::instance()->get_policy_handler();
    if(handler != nullptr)
    {
        id = handler->register_policy(when, f);
    }
    apex_policy_handle * handle = new apex_policy_handle();
    handle->id = id;
    handle->event_type = when;
    handle->period = 0;
    apex::instance()->push_policy_handle(handle);
    return handle;
}

std::set<apex_policy_handle*> register_policy(std::set<apex_event_type> when,
                    std::function<int(apex_context const&)> f)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return std::set<apex_policy_handle*>(); }
    std::set<apex_event_type>::iterator it;
    std::set<apex_policy_handle*> handles;
    for (it = when.begin(); it != when.end(); ++it)
    {
        handles.insert(register_policy(*it,f));
    }
    return handles;
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
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    int id = -1;
    policy_handler * handler = apex::instance()->get_policy_handler(period_microseconds);
    if(handler != nullptr)
    {
        id = handler->register_policy(APEX_PERIODIC, f);
    }
    apex_policy_handle * handle = new apex_policy_handle();
    handle->id = id;
    handle->event_type = APEX_PERIODIC;
    handle->period = period_microseconds;
    apex::instance()->push_policy_handle(handle);
    return handle;
}

#ifdef APEX_HAVE_HPX
int apex::setup_runtime_counter(const std::string & counter_name) {
    bool messaged = false;
    if(get_hpx_runtime_ptr() != nullptr) {
        using hpx::naming::id_type;
        using hpx::performance_counters::get_counter;
        using hpx::performance_counters::stubs::performance_counter;
        using hpx::performance_counters::counter_value;
        try {
            id_type id = get_counter(counter_name);
            if (id == hpx::naming::invalid_id) {
                if (instance()->get_node_id() == 0 && !messaged) {
                    std::cerr << "Error: invalid HPX counter: " << counter_name << std::endl;
                    messaged = true;
                }
                return APEX_ERROR;
            }
            performance_counter::start(hpx::launch::sync, id);
            registered_counters.emplace(std::make_pair(counter_name, id));
            //std::cout << "Started counter " << counter_name << std::endl;
        } catch(hpx::exception const & /*e*/) {
            if (instance()->get_node_id() == 0) {
                std::cerr << "Error: unable to start HPX counter: " << counter_name << std::endl;
            }
            return APEX_ERROR;
        }
    }
    return APEX_NOERROR;
}

void apex::query_runtime_counters(void) {
    if (instance()->get_node_id() > 0) {return;}
    using hpx::naming::id_type;
    using hpx::performance_counters::get_counter;
    using hpx::performance_counters::stubs::performance_counter;
    using hpx::performance_counters::counter_value;
    for (auto counter : registered_counters) {
        string name = counter.first;
        id_type id = counter.second;
        counter_value value1 = performance_counter::get_value(hpx::launch::sync, id);
        const int value = value1.get_value<int>();
        sample_value(name, value);
    }
}
#endif

apex_policy_handle * sample_runtime_counter(unsigned long period, const std::string & counter_name) {
    apex_policy_handle * handle = nullptr;
#ifdef APEX_HAVE_HPX
    if(get_hpx_runtime_ptr() != nullptr) {
        using hpx::naming::id_type;
        using hpx::performance_counters::get_counter;
        using hpx::performance_counters::stubs::performance_counter;
        using hpx::performance_counters::counter_value;
        id_type id = get_counter(counter_name);
        if (id == hpx::naming::invalid_id) {
            std::cerr << "Error: invalid HPX counter: " << counter_name << std::endl;
        }
        performance_counter::start(hpx::launch::sync, id);
        handle = register_periodic_policy(period, [=](apex_context const& ctx) -> int {
            try {
                counter_value value1 = performance_counter::get_value(hpx::launch::sync, id);
                const int value = value1.get_value<int>();
                sample_value(counter_name, value);
            } catch(hpx::exception const & /*e*/) {
                std::cerr << "Error: unable to start HPX counter: " << counter_name << std::endl;
                return APEX_ERROR;
            }
            return APEX_NOERROR;
        });
    }
#else
    std::cerr << "WARNING: Runtime counter sampling is not implemented for your runtime" << std::endl;
#endif
    return handle;
}

void deregister_policy(apex_policy_handle * handle) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // disable processing of policy for now
    //_notify_listeners = false;
    // if this policy has been deregistered already, return.
    if (!(apex::instance()->policy_handle_exists(handle))) { return; }
    policy_handler * handler = nullptr;
    if (handle->event_type == APEX_PERIODIC) {
        handler = apex::instance()->get_policy_handler(handle->period);
    } else {
        handler = apex::instance()->get_policy_handler();
    }
    if(handler != nullptr) {
        handler->deregister_policy(handle);
    }
    //_notify_listeners = true;
    apex::instance()->pop_policy_handle(handle);
    delete(handle);
}

apex_profile* get_profile(apex_function_address action_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    task_identifier id(action_address);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

apex_profile* get_profile(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    task_identifier id(timer_name);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

double current_power_high(void) {
    double power = 0.0;
#ifdef APEX_HAVE_RCR
    power = (double)rcr_current_power_high();
    //std::cout << "Read power from RCR: " << power << std::endl;
#elif APEX_HAVE_MSR
    power = msr_current_power_high();
    //std::cout << "Read power from MSR: " << power << std::endl;
#elif APEX_HAVE_PROC
    power = (double)read_power();
    //std::cout << "Read power from Cray Power Monitoring and Management: " << power << std::endl;
#else
    //std::cout << "NO POWER READING! Did you configure with RCR, MSR or Cray?" << std::endl;
#endif
    return power;
}

/*
std::vector<std::string> get_available_profiles() {
    return apex::__instance()->the_profiler_listener->get_available_profiles();
}
*/

void print_options() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    print_options();
    return;
}

void send (uint64_t tag, uint64_t size, uint64_t target) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return ; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return ; }
    // if APEX hasn't been initialized, do nothing.
    if (!_initialized) { return ; }
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) { return ; }

    if (_notify_listeners) {
        // eventually, we want to use the thread id, but for now, just use 0.
        //message_event_data data(tag, size, instance->get_node_id(), thread_instance::get_id(), target);
        message_event_data data(tag, size, instance->get_node_id(), 0, target);
        if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_send(data);
            }
        }
    }
}

void recv (uint64_t tag, uint64_t size, uint64_t source_rank, uint64_t source_thread) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return ; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return ; }
    // if APEX hasn't been initialized, do nothing.
    if (!_initialized) { return ; }
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) { return ; }

    if (_notify_listeners) {
        // eventually, we want to use the thread id, but for now, just use 0.
        //message_event_data data(tag, size, source_rank, source_thread, instance->get_node_id());
        message_event_data data(tag, size, source_rank, 0, instance->get_node_id());
        if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_recv(data);
            }
        }
    }
}

} // apex namespace

using namespace apex;

extern "C" {

    int apex_init(const char * thread_name, unsigned long int comm_rank, unsigned long int comm_size)
    {
        return init(thread_name, comm_rank, comm_size);
    }

    int apex_init_(unsigned long int comm_rank, unsigned long int comm_size) {
        return init("FORTRAN thread", comm_rank, comm_size);
    }

    int apex_init__(unsigned long int comm_rank, unsigned long int comm_size) {
        return init("FORTRAN thread", comm_rank, comm_size);
    }

    void apex_cleanup()
    {
        cleanup();
    }

    const char * apex_dump(bool reset)
    {
        return(dump(reset).c_str());
    }

    void apex_finalize()
    {
        finalize();
    }

    void apex_finalize_() { finalize(); }

    void apex_finalize__() { finalize(); }

    const char * apex_version()
    {
        return version().c_str();
    }

    apex_profiler_handle apex_start(apex_profiler_type type, void * identifier)
    {
      assert(identifier);
      if (type == APEX_FUNCTION_ADDRESS) {
          return reinterpret_cast<apex_profiler_handle>(start((apex_function_address)identifier));
      } else if (type == APEX_NAME_STRING) {
          string tmp((const char *)identifier);
          return reinterpret_cast<apex_profiler_handle>(start(tmp));
      }
      return APEX_NULL_PROFILER_HANDLE;
    }

    void apex_start_(apex_profiler_type type, void * identifier, apex_profiler_handle profiler) {
      apex_profiler_handle p = apex_start(type, identifier);
      if (profiler != nullptr) profiler = p;
    }

    apex_profiler_handle apex_resume(apex_profiler_type type, void * identifier)
    {
      assert(identifier);
      if (type == APEX_FUNCTION_ADDRESS) {
          return reinterpret_cast<apex_profiler_handle>(resume((apex_function_address)identifier));
      } else if (type == APEX_NAME_STRING) {
          string tmp((const char *)identifier);
          return reinterpret_cast<apex_profiler_handle>(resume(tmp));
      }
      return APEX_NULL_PROFILER_HANDLE;
    }

    void apex_reset(apex_profiler_type type, void * identifier) {
        if (type == APEX_FUNCTION_ADDRESS) {
            reset((apex_function_address)(identifier));
        } else {
            string tmp((const char *)identifier);
            reset(tmp);
        }
    }

    void apex_set_state(apex_thread_state state) {
        set_state(state);
    }

    void apex_stop(apex_profiler_handle the_profiler)
    {
        stop(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_stop_(apex_profiler_handle the_profiler)
    {
        stop(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_yield(apex_profiler_handle the_profiler)
    {
        yield(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_sample_value(const char * name, double value)
    {
        string tmp(name);
        sample_value(tmp, value);
    }

    void apex_new_task(apex_profiler_type type, void * identifier,
                       unsigned long long task_id) {
        if (type == APEX_FUNCTION_ADDRESS) {
            new_task((apex_function_address)(identifier), task_id);
        } else {
            string tmp((const char *)identifier);
            new_task(tmp, task_id);
        }
    }

    apex_event_type apex_register_custom_event(const char * name)
    {
        string tmp(name);
        return register_custom_event(tmp);
    }

    void apex_custom_event(apex_event_type event_type, void * custom_data)
    {
        custom_event(event_type, custom_data);
    }

    void apex_register_thread(const char * name)
    {
        if (name) {
            string tmp(name);
            register_thread(tmp);
        } else {
            string tmp("APEX WORKER THREAD");
            register_thread(tmp);
        }
    }

    void apex_exit_thread(void)
    {
        exit_thread();
    }

    apex_policy_handle* apex_register_policy(const apex_event_type when, int (f)(apex_context const)) {
        return register_policy(when, f);
    }

    apex_policy_handle* apex_register_periodic_policy(unsigned long period, int (f)(apex_context const)) {
        return register_periodic_policy(period, f);
    }

    void apex_deregister_policy(apex_policy_handle * handle) {
        return deregister_policy(handle);
    }

    apex_profile* apex_get_profile(apex_profiler_type type, void * identifier) {
        assert(identifier);
        if (type == APEX_FUNCTION_ADDRESS) {
            return get_profile((apex_function_address)(identifier));
        } else {
            string tmp((const char *)identifier);
            return get_profile(tmp);
        }
        return nullptr;
    }

    double apex_current_power_high() {
        return current_power_high();
    }

    void apex_print_options() {
        apex_options::print_options();
        return;
    }

    void apex_send (uint64_t tag, uint64_t size, uint64_t target) {
        return send(tag, size, target);
    }

    void apex_recv (uint64_t tag, uint64_t size, uint64_t source_rank, uint64_t source_thread) {
        return recv(tag, size, source_rank, source_thread);
    }

    uint64_t apex_hardware_concurrency(void) {
        return hardware_concurrency();
    }


} // extern "C"

#ifdef APEX_HAVE_HPX
HPX_DECLARE_ACTION(APEX_TOP_LEVEL_PACKAGE::finalize, apex_internal_shutdown_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(apex_internal_shutdown_action);
HPX_PLAIN_ACTION(APEX_TOP_LEVEL_PACKAGE::finalize, apex_internal_shutdown_action);

void apex_schedule_shutdown() {
    if(get_hpx_runtime_ptr() == nullptr) return;
    //if(!thread_instance::is_worker()) return;
    apex_internal_shutdown_action act;
    try {
        for(auto locality : hpx::find_all_localities()) {
            hpx::apply(act, locality);
        }
    } catch(...) {
        // what to do?
    }
}
#endif


