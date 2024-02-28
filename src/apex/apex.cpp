/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#include <hpx/modules/threading_base.hpp>
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
#include <atomic>
#include <set>
#include <sstream>
#include <vector>
#if APEX_WITH_PLUGINS
#include <dlfcn.h>
#endif
//#include <cxxabi.h> // this is for demangling strings.

#include "concurrency_handler.hpp"
#include "policy_handler.hpp"
#include "thread_instance.hpp"
#include "utils.hpp"
#include "apex_assert.h"
#include "event_filter.hpp"

#include "tau_listener.hpp"
#include "profiler_listener.hpp"
#include "trace_event_listener.hpp"
#if defined(APEX_WITH_PERFETTO)
#include "perfetto_listener.hpp"
#endif
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

#include "memory_wrapper.hpp"

#ifdef APEX_HAVE_HPX
#include <boost/assign.hpp>
#include <cstdint>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos_local/composable_guard.hpp>
#include "global_constructor_destructor.h"
#else // without HPX!
#ifdef APEX_USE_STATIC_GLOBAL_CONSTRUCTOR
#include "global_constructor_destructor.h"
#if defined(HAS_CONSTRUCTORS)
extern "C" {
DEFINE_CONSTRUCTOR(apex_init_static_void)
DEFINE_DESTRUCTOR(apex_finalize_static_void)
}
#endif // HAS_CONSTRUCTORS
#endif // APEX_USE_STATIC_GLOBAL_CONSTRUCTOR
#endif // APEX_HAVE_HPX

#ifdef APEX_HAVE_TCMALLOC
#include "tcmalloc_hooks.hpp"
#endif
#include "banner.hpp"
#include "apex_dynamic.hpp"

#if APEX_DEBUG
#define FUNCTION_ENTER if (apex_options::use_verbose()) { \
    fprintf(stderr, "enter %lu *** %s:%d!\n", \
    thread_instance::get_id(), __APEX_FUNCTION__, __LINE__); fflush(stdout); }
#define FUNCTION_EXIT if (apex_options::use_verbose()) { \
    fprintf(stderr, "exit  %lu *** %s:%d!\n", \
    thread_instance::get_id(), __APEX_FUNCTION__, __LINE__); fflush(stdout); }
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

bool& apex::get_program_over() {
    static bool _program_over{false};
    return _program_over;
}

// Global static pointer used to ensure a single instance of the class.
std::atomic<apex*> apex::m_pInstance(nullptr);

std::atomic<bool> _notify_listeners(true);
std::atomic<bool> _measurement_stopped(false);
/*
std::shared_ptr<task_wrapper>& top_level_timer() {
    static APEX_NATIVE_TLS std::shared_ptr<task_wrapper> top_level_timer = nullptr;
    return top_level_timer;
}
*/

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
        ss << "/threads{locality#" << instance->get_node_id() <<
            "/total}/count/cumulative";
        instance->setup_runtime_counter(ss.str());
*/
    }
}

#ifdef APEX_HAVE_HPX
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
    /*
    if (instance->get_node_id() == 0) {
        printf("APEX Finalizing...\n");
    }
    */
    // Shutdown APEX
    finalize();
    hpx_finalized = true;
    FUNCTION_EXIT
}
#endif // APEX_HAVE_HPX_disabled
#endif // APEX_HAVE_HPX

/*
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void apex::_initialize()
{
#if defined(APEX_DEBUG) || defined(APEX_ERROR_HANDLING)
    apex_register_signal_handler();
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
    tmp << APEX_VERSION_MAJOR << "."
        << APEX_VERSION_MINOR << "."
        << APEX_VERSION_PATCH;
#endif
#if defined (GIT_COMMIT_HASH)
    tmp << "-" << GIT_COMMIT_HASH ;
#endif
#if defined (GIT_BRANCH)
    tmp << "-" << GIT_BRANCH ;
#endif
    tmp << "\nBuilt on: " << __TIME__ << " " << __DATE__;
#if CMAKE_BUILD_TYPE == 1
    tmp << " (Release)";
#elif CMAKE_BUILD_TYPE == 2
    tmp << " (RelWithDebInfo)";
#else
    tmp << " (Debug)";
#endif
    tmp << "\nC++ Language Standard version : " << __cplusplus;
#if defined(__clang__)
    /* Clang/LLVM. ---------------------------------------------- */
    tmp << "\nClang Compiler version : " << __VERSION__;
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC. ------------------------------------------ */
    tmp << "\nIntel Compiler version : " << __VERSION__;
#elif defined(__GNUC__) || defined(__GNUG__)
    /* GNU GCC/G++. --------------------------------------------- */
    tmp << "\nGCC Compiler version : " << __VERSION__;
#elif defined(__HP_cc) || defined(__HP_aCC)
    /* Hewlett-Packard C/aC++. ---------------------------------- */
    tmp << "\nHP Compiler version : " << __HP_aCC;
#elif defined(__IBMC__) || defined(__IBMCPP__)
    /* IBM XL C/C++. -------------------------------------------- */
    tmp << "\nIBM Compiler version : " << __xlC__;
#elif defined(_MSC_VER)
    /* Microsoft Visual Studio. --------------------------------- */
    tmp << "\nMicrosoft Compiler version : " << _MSC_FULL_VER;
#elif defined(__PGI)
    /* Portland Group PGCC/PGCPP. ------------------------------- */
    tmp << "\nPGI Compiler version : " << __VERSION__;
#elif defined(__SUNPRO_CC)
    /* Oracle Solaris Studio. ----------------------------------- */
    tmp << "\nOracle Compiler version : " << __SUNPRO_CC;
#endif
    tmp << "\nConfigured features: Pthread";
#if defined(APEX_WITH_ACTIVEHARMONY) || defined(APEX_HAVE_ACTIVEHARMONY)
    tmp << ", AH";
#endif
#if defined(APEX_WITH_BFD) || defined(APEX_HAVE_BFD)
    tmp << ", BFD";
#endif
#if defined(APEX_WITH_CUDA) || defined(APEX_HAVE_CUDA)
    tmp << ", CUDA";
#endif
#if defined(APEX_WITH_HIP) || defined(APEX_HAVE_HIP)
    tmp << ", HIP";
#endif
#if defined(APEX_WITH_LEVEL0) || defined(APEX_HAVE_LEVEL0)
    tmp << ", L0";
#endif
#if defined(APEX_WITH_MPI) || defined(APEX_HAVE_MPI)
    tmp << ", MPI";
#endif
#if defined(APEX_WITH_OMPT) || defined(APEX_HAVE_OMPT)
    tmp << ", OMPT";
#endif
#if defined(APEX_WITH_OTF2) || defined(APEX_HAVE_OTF2)
    tmp << ", OTF2";
#endif
#if defined(APEX_WITH_PAPI) || defined(APEX_HAVE_PAPI)
    tmp << ", PAPI";
#endif
#if defined(APEX_WITH_PLUGINS) || defined(APEX_HAVE_PLUGINS)
    tmp << ", PLUGINS";
#endif
#if defined(APEX_WITH_STARPU) || defined(APEX_HAVE_STARPU)
    tmp << ", StarPU";
#endif
#if defined(APEX_WITH_PHIPROF) || defined(APEX_HAVE_PHIPROF)
    tmp << ", PhiProf";
#endif
#if defined(APEX_WITH_TCMALLOC) || defined(APEX_HAVE_TCMALLOC)
    tmp << ", TCMalloc";
#endif
#if defined(APEX_WITH_JEMALLOC) || defined(APEX_HAVE_JEMALLOC)
    tmp << ", JEMalloc";
#endif
#if defined(APEX_WITH_LM_SENSORS) || defined(APEX_HAVE_LM_SENSORS)
    tmp << ", LM Sensors";
#endif
#if defined(APEX_WITH_PERFETTO) || defined(APEX_HAVE_PERFETTO)
    tmp << ", Perfetto";
#endif
    tmp << "\n";

    this->version_string = std::string(tmp.str());
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
            the_otf2_listener = new otf2_listener();
            listeners.push_back(the_otf2_listener);
        }
#endif
#if defined(APEX_WITH_PERFETTO)
        if (apex_options::use_perfetto()) {
            the_perfetto_listener = new perfetto_listener();
            listeners.push_back(the_perfetto_listener);
        }
#endif
        if (apex_options::use_trace_event()) {
            the_trace_event_listener = new trace_event_listener();
            listeners.push_back(the_trace_event_listener);
        }

/* For the Jupyter support, always enable the concurrency handler. */
        if (apex_options::use_jupyter_support() ||
            apex_options::use_concurrency() > 0) {
            listeners.push_back(new
                concurrency_handler(apex_options::concurrency_period(),
            apex_options::use_concurrency()));
        }
        startup_throttling();
/* For the Jupyter support, always enable the policy listener. */
        if (apex_options::use_jupyter_support() ||
            apex_options::use_policy()) {
            this->m_policy_handler = new policy_handler();
            listeners.push_back(this->m_policy_handler);
        }
    }
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
    if (instance != nullptr && instance->the_profiler_listener != nullptr)
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

void do_atexit(void) {
    finalize();
    cleanup();
}

uint64_t init(const char * thread_name, uint64_t comm_rank,
    uint64_t comm_size) {
    FUNCTION_ENTER
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { FUNCTION_EXIT; return APEX_ERROR; }
    // FIRST! make sure APEX thinks this is a worker thread (the main thread
    // is always a worker thread)
    thread_instance::instance(true);
    // Just in case, if we got initialized without the correct rank and size,
    // Check the environment if there are MPI settings.
    if (comm_rank == 0 && comm_size == 1) {
        comm_rank = test_for_MPI_comm_rank(comm_rank);
        comm_size = test_for_MPI_comm_size(comm_size);
    }
    // protect against multiple initializations
    if (_registered || _initialized) {
        if (apex_options::use_jupyter_support()) {
            // reset all counters, and return.
            reset(APEX_NULL_FUNCTION_ADDRESS);
            FUNCTION_EXIT
            return APEX_NOERROR;
        } else {
            FUNCTION_EXIT
            return APEX_ERROR;
        }
    }
    /* register the finalization function, for program exit */
    std::atexit(do_atexit);
    //thread_instance::set_worker(true);
    _registered = true;
    apex* instance = apex::instance(); // get/create the Apex static instance
    // assign the rank and size.  Why not in the constructor?
    // because, if we registered a startup policy, the default
    // constructor was called, without the correct comm_rank and comm_size.
    if (comm_rank < comm_size && comm_size > 0) { // simple validation
      instance->set_node_id(comm_rank);
      instance->set_num_ranks(comm_size);
    }
    //printf("Node %lu of %lu\n", comm_rank, comm_size);
    if (!instance || _exited) {
        FUNCTION_EXIT
        return APEX_ERROR; // protect against calls after finalization
    }
    init_plugins();
    startup_event_data data(comm_rank, comm_size);
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(data);
        }
    }
    handle_delayed_start();
    // start accepting requests, now that all listeners are started
    _initialized = true;
    /* Note that because the PAPI support includes the ability to load and use
     * any configured components in PAPI, we need to make sure it's initialized
     * before we initialize any GPU support libraries (like cupti_init, or
     * hsa_init). */
#if APEX_HAVE_PROC
    if (apex_options::use_proc_cpuinfo() ||
        apex_options::use_proc_meminfo() ||
        apex_options::use_proc_net_dev() ||
        apex_options::use_proc_self_status() ||
        apex_options::monitor_gpu() ||
        apex_options::use_hip_profiler() ||
        apex_options::use_proc_stat() ||
        strlen(apex_options::papi_components()) > 0) {
        instance->pd_reader = new proc_data_reader();
    }
#endif
    /* for the next section, we need to check if we are suspended,
     * and if so, don't be suspended long enough to enable the main timer. */
    bool suspended = apex_options::suspend();
    apex_options::suspend(false);
    /* For the main thread, we should always start a top level timer.
     * The reason is that if the program calls "exit", our atexit() processing
     * will stop this timer, effectively stopping all of its children as well,
     * so we will get an accurate measurement for abnormal termination. */
    auto main = task_wrapper::get_apex_main_wrapper();
    // make sure the tracing support puts APEX MAIN on the right thread
    // when tracing HPX - the finalization will almost assuredly not
    // be stopped on the thread that is calling apex::init. You've been warned.
    main->explicit_trace_start = true;
    start(main);
    if (apex_options::top_level_os_threads()) {
        // start top-level timer for main thread, it will get automatically
        // stopped when the main wrapper timer is stopped.
        string task_name;
        if (thread_name) {
            stringstream ss;
            ss << "Main Thread: " << thread_name;
            task_name = ss.str();
        } else {
            task_name = "Main Thread";
        }
        std::shared_ptr<task_wrapper> twp =
            new_task(task_name, UINTMAX_MAX, main);
        start(twp);
        thread_instance::set_top_level_timer(twp);
    }
    /* restore the suspended bit */
    apex_options::suspend(suspended);
    if (apex_options::use_verbose() && instance->get_node_id() == 0) {
      std::cout << version() << std::endl;
      apex_options::print_options();
    }
    if (apex_options::throttle_energy() &&
        apex_options::throttle_concurrency() ) {
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
#ifdef APEX_HAVE_TCMALLOC
    //tcmalloc::init_hook();
    enable_memory_wrapper();
#else
    enable_memory_wrapper();
#endif
    if (apex_options::delay_memory_tracking()) {
        if (instance->get_node_id() == 0) {
            std::cout << "Pausing memory tracking until further notice..." << std::endl;
        }
        controlMemoryWrapper(false);
    } else {
        if (instance->get_node_id() == 0) {
            std::cout << "Enabling memory tracking!" << std::endl;
        }
        controlMemoryWrapper(true);
    }

    // It's now safe to initialize CUDA and/or HIP and/or Level0
    dynamic::cuda::init();
    dynamic::roctracer::init();
    dynamic::level0::init();

    // Unset the LD_PRELOAD variable, because Active Harmony is going to
    // fork/execv a new session-core process, and we don't want APEX in
    // that forked process.
    const char * preload = getenv("LD_PRELOAD");
    if (preload != nullptr) {
        unsetenv("LD_PRELOAD");
    }
    if (comm_rank == 0) {
        printf("%s", apex_banner);
        printf("APEX Version: %s\n", instance->version_string.c_str());
    }
    FUNCTION_EXIT
    return APEX_NOERROR;
}

string& version() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        static string tmp("disabled"); return tmp;
    }
    apex* instance = apex::instance(); // get the Apex static instance
    return instance->version_string;
}

/* Populate the new task_wrapper object, and notify listeners. */
inline std::shared_ptr<task_wrapper> _new_task(
    task_identifier * id,
    const uint64_t task_id,
    const std::shared_ptr<task_wrapper> parent_task, apex* instance) {
    in_apex prevent_deadlocks;
    APEX_UNUSED(instance);
    std::shared_ptr<task_wrapper> tt_ptr = make_shared<task_wrapper>();
    tt_ptr->task_id = id;
    // get the thread id that is creating this task
    tt_ptr->thread_id = thread_instance::instance().get_id();
    // if not tracking dependencies, don't save the parent
    /* Why?
    if ((!apex_options::use_taskgraph_output()) &&
         !apex_options::use_otf2()) {
        tt_ptr->parent = task_wrapper::get_apex_main_wrapper();
    // was a parent passed in?
    } else */ if (parent_task != nullptr) {
        tt_ptr->parent_guid = parent_task->guid;
        tt_ptr->parent = parent_task;
    // if not, is there a current timer?
    } else {
        profiler * p = thread_instance::instance().get_current_profiler();
        if (p != nullptr) {
            tt_ptr->parent_guid = p->guid;
            tt_ptr->parent = p->tt_ptr;
        } else {
            tt_ptr->parent = task_wrapper::get_apex_main_wrapper();
        }
    }
    if (apex_options::use_tasktree_output() || apex_options::use_hatchet_output()) {
        tt_ptr->assign_heritage();
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

void debug_print(const char * event, std::shared_ptr<task_wrapper> tt_ptr) {
    if (apex::get_program_over()) return;
    static std::mutex this_mutex;
    std::unique_lock<std::mutex> l(this_mutex);
    std::stringstream ss;
    if (tt_ptr == nullptr) {
        ss << thread_instance::get_id() << " " << event << " : (null) : (null)"
            << endl;
        cout << ss.str(); fflush(stdout);
    } else {
        ss << thread_instance::get_id() << " " << event << " : " <<
            tt_ptr->guid << " : " << tt_ptr->get_task_id()->get_name() << endl;
        cout << ss.str(); fflush(stdout);
    }
}

profiler* start(const std::string &timer_name)
{
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_DISABLED_START
        return nullptr;
    }
    //printf("%lu: %s\n", thread_instance::get_id(), timer_name.c_str());
    //fflush(stdout);
    const std::string apex_internal("apex_internal");
    if (starts_with(timer_name, apex_internal)) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
        // don't process our own events - queue scrubbing tasks.
        return profiler::get_disabled_profiler();
    }
    // don't time filtered events
    if (event_filter::instance().have_filter && event_filter::exclude(timer_name)) {
        return profiler::get_disabled_profiler();
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
    profiler * new_profiler = nullptr;
    if (_notify_listeners) {
        bool success = true;
        task_identifier * id = task_identifier::get_task_id(timer_name);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
#if defined(APEX_DEBUG)//_disabled)
        if (apex_options::use_verbose()) { debug_print("Start", tt_ptr); }
#endif
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        //read_lock_type l(instance->listener_mutex);
        /*
        std::stringstream dbg;
        dbg << thread_instance::get_id() << " Start : " << id->get_name() << endl;
            printf("%s\n",dbg.str().c_str());
        fflush(stdout);
        */
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            tt_ptr->prof = thread_instance::instance().get_current_profiler();
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " <<
                //id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                return profiler::get_disabled_profiler();
            }
        }
        // If we are allowing untied timers, clear the timer stack on this thread
        if (apex_options::untied_timers() == true) {
            new_profiler = thread_instance::instance().get_current_profiler();
            thread_instance::instance().clear_current_profiler();
        }
    }
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (timer_name.compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
    } else {
        APEX_UTIL_REF_COUNT_START
    }
#endif
    if (apex_options::untied_timers() == true) {
        return new_profiler;
    }
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* start(const apex_function_address function_address) {
    in_apex prevent_deadlocks;
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
    profiler * new_profiler = nullptr;
    if (_notify_listeners) {
        bool success = true;
        task_identifier * id = task_identifier::get_task_id(function_address);
        tt_ptr = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
#if defined(APEX_DEBUG)//_disabled)
        if (apex_options::use_verbose()) { debug_print("Start", tt_ptr); }
#endif
        APEX_UTIL_REF_COUNT_TASK_WRAPPER
        /*
        std::stringstream dbg;
        dbg << thread_instance::get_id() << " Start : " << id->get_name() << endl;
            printf("%s\n",dbg.str().c_str());
        fflush(stdout);
        */
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            tt_ptr->prof = thread_instance::instance().get_current_profiler();
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " <<
                //id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                return profiler::get_disabled_profiler();
            }
        }
        // If we are allowing untied timers, clear the timer stack on this thread
        if (apex_options::untied_timers() == true) {
            new_profiler = thread_instance::instance().get_current_profiler();
            thread_instance::instance().clear_current_profiler();
        }
    }
    APEX_UTIL_REF_COUNT_START
    if (apex_options::untied_timers() == true) {
        return new_profiler;
    }
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

void start(std::shared_ptr<task_wrapper> tt_ptr) {
    in_apex prevent_deadlocks;
#if defined(APEX_DEBUG)//_disabled)
    if (apex_options::use_verbose()) { debug_print("Start", tt_ptr); }
#endif
    if (tt_ptr == nullptr) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
        return;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_DISABLED_START
        tt_ptr->prof = nullptr;
        return;
    }
    // don't time filtered events
    if (event_filter::instance().have_filter && event_filter::exclude(tt_ptr->task_id->get_name())) {
        tt_ptr->prof = nullptr;
        return;
    }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE
        tt_ptr->prof = nullptr;
        return;
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) {
        APEX_UTIL_REF_COUNT_SUSPENDED_START
        tt_ptr->prof = profiler::get_disabled_profiler();
        return;
    }
    // get the thread id that is running this task
    tt_ptr->thread_id = thread_instance::instance().get_id();
    if (_notify_listeners) {
        bool success = true;
        /*
        std::stringstream dbg;
        dbg << thread_instance::get_id() << " Start : " << tt_ptr->task_id->get_name() << endl;
        printf("%s\n",dbg.str().c_str());
        fflush(stdout);
        */
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(tt_ptr);
            tt_ptr->prof = thread_instance::instance().get_current_profiler();
            if (!success && i == 0) {
                //cout << thread_instance::get_id() << " *** Not success! " <<
                //id->get_name() << endl; fflush(stdout);
                APEX_UTIL_REF_COUNT_FAILED_START
                tt_ptr->prof = profiler::get_disabled_profiler();
                return;
            }
        }
        // If we are allowing untied timers, clear the timer stack on this thread
        if (apex_options::untied_timers() == true) {
            thread_instance::instance().clear_current_profiler();
        }
    }
    APEX_UTIL_REF_COUNT_START
    thread_instance::instance().restore_children_profilers(tt_ptr);
    return;
}

profiler* resume(const std::string &timer_name) {
    in_apex prevent_deadlocks;
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
    // don't time filtered events
    if (event_filter::instance().have_filter && event_filter::exclude(timer_name)) {
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
        } catch (disabled_profiler_exception &e) {
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler();
        }
    }
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (timer_name.compare(apex_process_profile_str) == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
    } else {
        APEX_UTIL_REF_COUNT_RESUME
    }
#endif
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* resume(const apex_function_address function_address) {
    in_apex prevent_deadlocks;
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
        } catch (disabled_profiler_exception &e) {
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler();
        }
    }
    APEX_UTIL_REF_COUNT_RESUME
    return thread_instance::instance().restore_children_profilers(tt_ptr);
}

profiler* resume(profiler * p) {
    in_apex prevent_deadlocks;
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
        } catch (disabled_profiler_exception &e) {
            APEX_UTIL_REF_COUNT_FAILED_RESUME
            return profiler::get_disabled_profiler();
        }
    }
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
    } else {
        APEX_UTIL_REF_COUNT_RESUME
    }
#endif
    return p;
}

void reset(const std::string &timer_name) {
    in_apex prevent_deadlocks;
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
    in_apex prevent_deadlocks;
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

void apex::complete_task(std::shared_ptr<task_wrapper> task_wrapper_ptr) {
    apex* instance = apex::instance(); // get the Apex static instance
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_task_complete(task_wrapper_ptr);
        }
    }
}

void apex::stop_internal(profiler* the_profiler) {
    in_apex prevent_deadlocks;
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
#if defined(APEX_DEBUG)//_disabled)
    if (apex_options::use_verbose()) { debug_print("Stop", the_profiler->tt_ptr); }
#endif
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
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
    } else {
        APEX_UTIL_REF_COUNT_STOP
    }
#endif
    instance->complete_task(p->tt_ptr);
    p->tt_ptr = nullptr;
}

void stop(profiler* the_profiler, bool cleanup) {
    in_apex prevent_deadlocks;
    // protect against calls after finalization
    if (_exited || _measurement_stopped) {
        APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE
        return;
    }
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
#if defined(APEX_DEBUG)//_disabled)
    if (apex_options::use_verbose()) { debug_print("Stop", the_profiler->tt_ptr); }
#endif
    thread_instance::instance().clear_current_profiler(the_profiler, false,
        null_task_wrapper);
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
    /*
    std::stringstream dbg;
    dbg << thread_instance::get_id() << "->" << the_profiler->tt_ptr->thread_id << " Stop : " <<
    the_profiler->tt_ptr->get_task_id()->get_name() << endl;
            printf("%s\n",dbg.str().c_str());
    fflush(stdout);
    */
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
    } else {
        APEX_UTIL_REF_COUNT_STOP
    }
#endif
    if (cleanup) {
        instance->complete_task(p->tt_ptr);
        //instance->active_task_wrappers.erase(p->tt_ptr);
        p->tt_ptr = nullptr;
    }
}

void stop(std::shared_ptr<task_wrapper> tt_ptr) {
    in_apex prevent_deadlocks;
    // protect against calls after finalization
    if (_exited || _measurement_stopped) {
        APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE
        return;
    }
#if defined(APEX_DEBUG)//_disabled)
    if (apex_options::use_verbose()) { debug_print("Stop", tt_ptr); }
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
    thread_instance::instance().clear_current_profiler(tt_ptr->prof, false,
        null_task_wrapper);
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
    /*
    std::stringstream dbg;
    dbg << thread_instance::get_id() << "->" << tt_ptr->thread_id << " Stop : " <<
    tt_ptr->get_task_id()->get_name() << endl;
            printf("%s\n",dbg.str().c_str());
    fflush(stdout);
    */
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
    } else {
        APEX_UTIL_REF_COUNT_STOP
    }
#endif
    instance->complete_task(tt_ptr);
}

void yield(profiler* the_profiler)
{
    in_apex prevent_deadlocks;
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
    thread_instance::instance().clear_current_profiler(the_profiler, false,
        null_task_wrapper);
    std::shared_ptr<profiler> p{the_profiler};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_yield(p);
        }
    }
    //cout << thread_instance::get_id() << " Yield : " <<
    //the_profiler->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD
    } else {
        APEX_UTIL_REF_COUNT_YIELD
    }
#endif
}

void yield(std::shared_ptr<task_wrapper> tt_ptr)
{
    in_apex prevent_deadlocks;
#if defined(APEX_DEBUG)//_disabled)
    if (apex_options::use_verbose()) { debug_print("Yield", tt_ptr); }
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
    thread_instance::instance().clear_current_profiler(tt_ptr->prof,
        true, tt_ptr);
    std::shared_ptr<profiler> p{tt_ptr->prof};
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_yield(p);
        }
    }
    //cout << thread_instance::get_id() << " Yield : " <<
    //tt_ptr->prof->tt_ptr->get_task_id()->get_name() << endl; fflush(stdout);
#if defined(APEX_DEBUG)
    const std::string apex_process_profile_str("apex::process_profiles");
    if (p->tt_ptr->get_task_id()->get_name(false).compare(apex_process_profile_str)
        == 0) {
        APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD
    } else {
        APEX_UTIL_REF_COUNT_YIELD
    }
#endif
    tt_ptr->prof = nullptr;
}

void sample_value(const std::string &name, double value, bool threaded)
{
    in_apex prevent_deadlocks;
    // check these before checking the options, because if we have already
    // cleaned up, checking the options can cause deadlock. This can
    // happen if we are tracking memory.
    if (_exited || _measurement_stopped) return; // protect against calls after finalization
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
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
              if (strstr(token, "worker-thread")==nullptr) { break; }
              token = strtok(nullptr, "/");
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
    sample_value_event_data data(tid, name, value, threaded);
    if (_notify_listeners) {
        //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_sample_value(data);
        }
    }
}

std::shared_ptr<task_wrapper> new_task(
    const std::string &name,
    const uint64_t task_id,
    const std::shared_ptr<task_wrapper> parent_task)
{
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr;
    }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr;
    }
    const std::string apex_internal("apex_internal");
    if (starts_with(name, apex_internal)) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        // don't process our own events - queue scrubbing tasks.
        return nullptr;
    }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr;
    } // protect against calls after finalization
    task_identifier * id = task_identifier::get_task_id(name);
    std::shared_ptr<task_wrapper>
        tt_ptr(_new_task(id, task_id, parent_task, instance));
    APEX_UTIL_REF_COUNT_TASK_WRAPPER
    return tt_ptr;
}

std::shared_ptr<task_wrapper> new_task(
    const apex_function_address function_address,
    const uint64_t task_id,
    const std::shared_ptr<task_wrapper> parent_task) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    task_identifier * id = task_identifier::get_task_id(function_address);
    std::shared_ptr<task_wrapper>
        tt_ptr(_new_task(id, task_id, parent_task, instance));
    return tt_ptr;
}

std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> wrapper,
    const std::string &timer_name) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    if (wrapper == nullptr) {
        // get the Apex static instance
        apex* instance = apex::instance();
        // protect against calls after finalization
        if (!instance || _exited) { return nullptr; }
        task_identifier * id = task_identifier::get_task_id(timer_name);
        wrapper = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
    } else {
        task_identifier * id = task_identifier::get_task_id(timer_name);
        // only have to do something if the ID has changed
        if (id != wrapper->get_task_id()) {
            // If a profiler was already started, yield it and start a new one with the new ID
            if (wrapper->prof != nullptr) {
                yield(wrapper);
                //wrapper->prof->set_task_id(wrapper->get_task_id());
                wrapper->alias = id;
                start(wrapper);
            } else {
                wrapper->alias = id;
            }
            if (apex_options::use_tasktree_output() || apex_options::use_hatchet_output()) {
                wrapper->update_heritage();
            }
        /*
        printf("%llu New alias: %s to %s\n", wrapper->guid,
           wrapper->task_id->get_name().c_str(), timer_name.c_str());
           */
        }
    }
    return wrapper;
}

std::shared_ptr<task_wrapper> update_task(
    std::shared_ptr<task_wrapper> wrapper,
    const apex_function_address function_address) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) {
        APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER
        return nullptr; }
    if (wrapper == nullptr) {
        // get the Apex static instance
        apex* instance = apex::instance();
        // protect against calls after finalization
        if (!instance || _exited) { return nullptr; }
        task_identifier * id = task_identifier::get_task_id(function_address);
        wrapper = _new_task(id, UINTMAX_MAX, null_task_wrapper, instance);
    } else {
        task_identifier * id = task_identifier::get_task_id(function_address);
        // only have to do something if the ID has changed
        if (id != wrapper->get_task_id()) {
            if (wrapper->prof != nullptr) {
                yield(wrapper);
                wrapper->alias = id;
                //wrapper->prof->set_task_id(wrapper->get_task_id());
                start(wrapper);
            } else {
                wrapper->alias = id;
            }
            if (apex_options::use_tasktree_output() || apex_options::use_hatchet_output()) {
                wrapper->update_heritage();
            }
        }
    }
    return wrapper;
}

std::atomic<int> custom_event_count(APEX_CUSTOM_EVENT_1);

apex_event_type register_custom_event(const std::string &name) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return APEX_CUSTOM_EVENT_1; }
    apex* instance = apex::instance(); // get the Apex static instance
    // protect against calls after finalization
    if (!instance || _exited) return APEX_CUSTOM_EVENT_1;
    if (custom_event_count == APEX_MAX_EVENTS) {
      std::cerr << "Cannot register more than MAX Events! (set to " <<
      APEX_MAX_EVENTS << ")" << std::endl;
    }
    write_lock_type l(instance->custom_event_mutex);
    instance->custom_event_names[custom_event_count] = name;
    int tmp = custom_event_count;
    custom_event_count++;
    return (apex_event_type)tmp;
}

void custom_event(apex_event_type event_type, void * custom_data) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) { return; }
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
#ifdef APEX_WITH_PLUGINS
    FUNCTION_ENTER
    if (apex_options::disable() == true) { return; }
    std::string plugin_names_str{apex_options::plugins()};
    std::string plugins_prefix{apex_options::plugins_path()};
#if defined(__APPLE__)
    std::string plugins_suffix{".dylib"};
#else
    std::string plugins_suffix{".so"};
#endif
    if(plugin_names_str.empty()) {
        FUNCTION_EXIT
        return;
    }
    std::vector<std::string> plugin_names;
    std::vector<std::string> plugin_paths;
    split( plugin_names_str, ':', plugin_names);
    for(const std::string & plugin_name : plugin_names) {
        plugin_paths.push_back(plugins_prefix + "/" + plugin_name +
        plugins_suffix);
    }
    for(const std::string & plugin_path : plugin_paths) {
        const char * path = plugin_path.c_str();
        void * plugin_handle = dlopen(path, RTLD_NOW);
        if(!plugin_handle) {
            std::cerr << "Error loading plugin " << path << ": " << dlerror()
            << std::endl;
            FUNCTION_EXIT
            continue;
        }
        int (*init_fn)() = (int (*)()) ((uintptr_t) dlsym(plugin_handle,
            "apex_plugin_init"));
        if(!init_fn) {
            std::cerr << "Error loading apex_plugin_init from " << path << ": "
                << dlerror() << std::endl;
            dlclose(plugin_handle);
            FUNCTION_EXIT
            continue;
        }
        int (*finalize_fn)() = (int (*)()) ((uintptr_t) dlsym(plugin_handle,
            "apex_plugin_finalize"));
        if(!finalize_fn) {
            std::cerr << "Error loading apex_plugin_finalize from " << path <<
                ": " << dlerror() << std::endl;
            dlclose(plugin_handle);
            FUNCTION_EXIT
            continue;
        }
        apex * instance = apex::instance();
        if(!instance) {
            std::cerr << "Error getting APEX instance while registering"
                << " finalize function from " << path << std::endl;
            FUNCTION_EXIT
            continue;
        }
        instance->finalize_functions.push_back(finalize_fn);
        int result = init_fn();
        if(result != 0) {
            std::cerr << "Error: apex_plugin_init for " << path << " returned "
                << result << std::endl;
            dlclose(plugin_handle);
            FUNCTION_EXIT
            continue;
        }
    }
    FUNCTION_EXIT
#endif
}

void finalize_plugins(void) {
#ifdef APEX_WITH_PLUGINS
    FUNCTION_ENTER
    if (apex_options::disable() == true) { return; }
    apex * instance = apex::instance();
    if(!instance) return;
    for(int (*finalize_function)() : instance->finalize_functions) {
        int result = finalize_function();
        if(result != 0) {
            std::cerr << "Error: plugin finalize function returned " << result
                << std::endl;
            continue;
        }
    }
    FUNCTION_EXIT
#endif
}

std::string dump(bool reset, bool finalizing) {
    in_apex prevent_deadlocks;
    static int index{0};
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true ||
        (!finalizing && apex_options::use_final_output_only()))
            { return(std::string("")); }
    bool old_screen_output = apex_options::use_screen_output();
    if (apex_options::use_jupyter_support()) {
        // force output in the Jupyter notebook
        apex_options::use_screen_output(true);
    }

    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance) { FUNCTION_EXIT return(std::string("")); }
    dynamic::cuda::flush();
    dynamic::roctracer::flush();
    dynamic::level0::flush();
    /* only track after N calls to apex::dump() */
    index = index + 1;
    if (apex_options::delay_memory_tracking() &&
        index > apex_options::delay_memory_iterations()) {
        if (instance->get_node_id() == 0) {
            std::cout << "Enabling memory tracking!" << std::endl;
        }
        controlMemoryWrapper(true);
    } 
    if (_notify_listeners) {
        //apex_get_leak_symbols();
        dump_event_data data(instance->get_node_id(),
            thread_instance::get_id(), reset);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_dump(data);
        }
        if (apex_options::use_jupyter_support()) {
            apex_options::use_screen_output(old_screen_output);
        }
        return(data.output);
    }
    if (apex_options::use_jupyter_support()) {
        apex_options::use_screen_output(old_screen_output);
    }
    return(std::string(""));
}

void finalize(void)
{
    in_apex prevent_deadlocks;
    if (!_initialized) { return; } // protect against finalization without initialization
    // prevent re-entry, be extra strict about race conditions - it is
    // possible.
    mutex shutdown_mutex;
    static bool finalized = false;
    {
        unique_lock<mutex> l(shutdown_mutex);
        if (finalized) { return; };
        finalized = true;
    }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) { return; } // protect against calls after finalization
    if (apex_options::use_jupyter_support()) {
        // reset all counters, and return.
        reset(APEX_NULL_FUNCTION_ADDRESS);
        return;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    FUNCTION_ENTER
    instance->finalizing = true; // don't measure any new pthreads from pthread_create!
    // FIRST FIRST, check if we have orphaned threads...
    // See apex::register_thread and apex::exit_thread for more info.
    /* this causes problems with APPLE, but that's ok because it's mostly
     * a problem with linux threaded runtimes when the main thread exits
     * before the worker threads have finished */
#if !defined(__APPLE__)
    {
        std::unique_lock<std::mutex> l(instance->thread_instance_mutex);
        if (!instance->known_threads.empty()) {
            for (thread_instance* t : instance->known_threads) {
                // make sure it hasn't been erased!
                if (instance->erased_threads.find(t) ==
                    instance->erased_threads.end()) {
                    t->clear_all_profilers();
                }
            }
        }
    }
#endif
    // FIRST, stop the top level timer, while the infrastructure is still
    // functioning.
    auto tmp = thread_instance::get_top_level_timer();
    if (tmp != nullptr) {
        stop(tmp);
        thread_instance::clear_top_level_timer();
    }
    // Second, stop the main timer, while the infrastructure is still
    // functioning.
    instance->the_profiler_listener->stop_main_timer();
    // if not done already...
    shutdown_throttling(); // stop thread scheduler policies
    /* Do this before OTF2 grabs a final timestamp - we might have
     * to terminate some OMPT events. */
    dynamic::ompt::do_shutdown();
    // notify all listeners that we are going to stop soon
    for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
        instance->listeners[i]->on_pre_shutdown();
    }
    stop_all_async_threads(); // stop OS/HW monitoring, including PAPI

    /* This could take a while */
    dynamic::cuda::flush();
    dynamic::roctracer::flush();
    dynamic::level0::flush();

    // stop processing new timers/counters/messages/tasks/etc.
    dynamic::cuda::stop();
    dynamic::roctracer::stop();
    dynamic::level0::stop();
    apex_options::suspend(true);

    // now, process all output
    dump(false, true);
    exit_thread();
    if (!_measurement_stopped)
    {
        _measurement_stopped = true;
        APEX_UTIL_REPORT_STATS
#ifdef APEX_HAVE_HPX
        /* HPX shutdown happens on a new thread. We don't want
         * to register a new thread. */
        shutdown_event_data data(instance->get_node_id(), 0);
#else
        shutdown_event_data data(instance->get_node_id(),
            thread_instance::get_id());
#endif
        _notify_listeners = false;
        {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_shutdown(data);
            }
        }
    }
#ifdef APEX_HAVE_TCMALLOC
    //tcmalloc::destroy_hook();
#endif
    disable_memory_wrapper();
    apex_report_leaks();
#if APEX_HAVE_BFD
    address_resolution::delete_instance();
#endif
    FUNCTION_EXIT
}

void cleanup(void) {
    in_apex prevent_deadlocks;
    FUNCTION_ENTER
    apex::get_program_over() = true;
#ifdef APEX_HAVE_HPX
    // prevent crash at shutdown.
    return;
#endif
    if (apex_options::use_jupyter_support()) {
        apex_options::use_jupyter_support(false);
        finalize();
    }
    // prevent re-entry, be extra strict about race conditions - it is
    // possible.
    mutex shutdown_mutex;
    static bool finalized = false;
    {
        unique_lock<mutex> l(shutdown_mutex);
        if (finalized) { return; };
        finalized = true;
    }
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { FUNCTION_EXIT return; }
    // get the Apex static instance
    apex* instance = apex::__instance();
    // protect against multiple calls
    if (!instance) { FUNCTION_EXIT return; }
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

void register_thread_hpx(const std::string &name) {
    register_thread(name);
}

void register_thread(const std::string &name,
    std::shared_ptr<task_wrapper> parent)
{
    in_apex prevent_deadlocks;
    FUNCTION_ENTER
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) return;
    // protect against multiple registrations on the same thread
    if (_registered) return;
    _registered = true;
    // FIRST! make sure APEX thinks this is a worker thread
    auto& ti = thread_instance::instance(true);
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
        std::string task_name;
        if (name.find("worker-thread") != name.npos) {
            ss << "worker-thread";
            task_name = ss.str();
        } else {
            string::size_type index = name.find("#");
            if (index!=std::string::npos) {
                string short_name = name.substr(0,index);
                ss << short_name;
            } else {
                ss << name;
            }
            task_name = ss.str();
        }
        // if the parent is null, then assume the new thread was
        // spawned by the main timer.
        std::shared_ptr<task_wrapper> twp =
            new_task(task_name, UINTMAX_MAX,
            (parent == nullptr ? task_wrapper::get_apex_main_wrapper() : parent));
        start(twp);
        //printf("New thread: %p\n", &(*twp));
        thread_instance::set_top_level_timer(twp);
    }
    /* What is this about? Well, the pthread_create wrapper in APEX can
     * capture pthreads that are spawned from libraries before main is
     * executed! If that is the case, then we want to (dangerously)
     * store the thread_instance pointer (which is a thread_local variable)
     * and at exit, if the thread hasn't been exited we want to stop
     * its timers (because likely it will exit when the library's destructor
     * is called. So we store any threads that are seen by the wrapper. */
    std::string prefix{"APEX pthread wrapper"};
    if (name.substr(0, prefix.size()) == prefix) {
        std::unique_lock<std::mutex> l{instance->thread_instance_mutex};
        instance->known_threads.insert(&ti);
    }
}

void exit_thread(void)
{
    in_apex prevent_deadlocks;
    // get the Apex static instance
    apex* instance = apex::instance();
    // protect against calls after finalization
    if (!instance || _exited) return;
    // protect against multiple calls from the same thread
    static APEX_NATIVE_TLS bool _exiting = false;
    if (_exiting) return;
    _exiting = true;
    {
        /* What's this about? see what happens when threads are registered.
         * When threads exit, we want to remove them from the known_thread
         * set, if we saved them. */
        thread_instance& ti = thread_instance::instance(false);
        std::unique_lock<std::mutex> l{instance->thread_instance_mutex};
        if (instance->known_threads.find(&ti) != instance->known_threads.end()) {
            instance->erased_threads.insert(&ti);
            instance->known_threads.erase(&ti);
        }
    }
    auto tmp = thread_instance::get_top_level_timer();
    // tell the timer cleanup that we are exiting
    thread_instance::exiting();
    //printf("Old thread: %p\n", &(*tmp));
    if (tmp != nullptr) {
        stop(tmp);
        thread_instance::clear_top_level_timer();
    }
    // ok to set this now - we need everything still running
    _exited = true;
    event_data data;
    if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_exit_thread(data);
        }
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
    in_apex prevent_deadlocks;
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
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) {
        return std::set<apex_policy_handle*>();
    }
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
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    int id = -1;
    policy_handler * handler =
        apex::instance()->get_policy_handler(period_microseconds);
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
        using hpx::performance_counters::performance_counter;
        using hpx::performance_counters::counter_value;
        try {
            performance_counter counter(counter_name);
            /*
            if (id == hpx::naming::invalid_id) {
                if (instance()->get_node_id() == 0 && !messaged) {
                    std::cerr << "Error: invalid HPX counter: " << counter_name
                        << std::endl;
                    messaged = true;
                }
                return APEX_ERROR;
            }*/
            counter.start(hpx::launch::sync);
            registered_counters.emplace(std::make_pair(counter_name, counter));
            //std::cout << "Started counter " << counter_name << std::endl;
        } catch(hpx::exception const & /*e*/) {
            if (instance()->get_node_id() == 0) {
                std::cerr << "Error: unable to start HPX counter: " <<
                    counter_name << std::endl;
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
    using hpx::performance_counters::performance_counter;
    using hpx::performance_counters::counter_value;
    for (auto counter : registered_counters) {
        string name = counter.first;
        performance_counter c = counter.second;
        counter_value cv = c.get_counter_value(hpx::launch::sync);
        const int value = cv.get_value<int>();
        sample_value(name, value);
    }
}
#endif

apex_policy_handle * sample_runtime_counter(unsigned long period, const
    std::string & counter_name) {
    apex_policy_handle * handle = nullptr;
#ifdef APEX_HAVE_HPX
    if(get_hpx_runtime_ptr() != nullptr) {
        using hpx::naming::id_type;
        using hpx::performance_counters::get_counter;
        using hpx::performance_counters::performance_counter;
        using hpx::performance_counters::counter_value;
        performance_counter counter(counter_name);
        /*
        if (id == hpx::naming::invalid_id) {
            std::cerr << "Error: invalid HPX counter: " << counter_name <<
                std::endl;
        }
        */
        counter.start(hpx::launch::sync);
        handle = register_periodic_policy(period,
                 [=](apex_context const& ctx) -> int {
            try {
                counter_value cv = counter.get_counter_value(hpx::launch::sync);
                const int value = cv.get_value<int>();
                sample_value(counter_name, value);
            } catch(hpx::exception const & /*e*/) {
                std::cerr << "Error: unable to start HPX counter: " <<
                    counter_name << std::endl;
                return APEX_ERROR;
            }
            return APEX_NOERROR;
        });
    }
#else
    APEX_UNUSED(period);
    APEX_UNUSED(counter_name);
    std::cerr <<
        "WARNING: Runtime counter sampling is not implemented for your runtime"
        << std::endl;
#endif
    return handle;
}

void deregister_policy(apex_policy_handle * handle) {
    in_apex prevent_deadlocks;
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

void stop_all_async_threads(void) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) { FUNCTION_EXIT return; } // protect against calls after finalization
    finalize_plugins();
    instance->stop_all_policy_handles();
#if APEX_HAVE_PROC
    if (instance->pd_reader != nullptr) {
        instance->pd_reader->stop_reading();
    }
#endif
#if APEX_HAVE_MSR
    apex_finalize_msr();
#endif
}

apex_profile* get_profile(apex_function_address action_address) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    task_identifier id(action_address);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

apex_profile* get_profile(const std::string &timer_name) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    task_identifier id(timer_name);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

apex_profile* get_profile(const task_identifier &task_id) {
    in_apex prevent_deadlocks;
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(task_id);
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
#elif APEX_HAVE_POWERCAP_POWER
    static long long lpower = 0LL;
    long long tpower = 0LL;
    static long long diff = 0LL;
    tpower = read_package0(false);
    /* Check for overflow */
    if (lpower > 0 && tpower > lpower) {
        diff = tpower-lpower;
    } else {
        diff = 0;
    }
    power = (double)diff;
    lpower = tpower;
    // convert from microjoules to joules
    power = power * 1.0e-6;
    // get the right joules per second
    power = power * (1.0e6 / (double)apex_options::concurrency_period());
    //std::cout << "Read power from Powercap: " << tpower << " " << lpower << " " << diff << " " << power << std::endl;
#elif APEX_HAVE_PROC
    power = (double)read_power();
    //std::cout << "Read power from Cray Power Monitoring and Management: " <<
#else
    std::cout << "NO POWER READING! Did you configure with RCR, MSR or Cray?"
    << std::endl;
#endif
    return power;
}

std::vector<task_identifier>& get_available_profiles() {
    return apex::__instance()->the_profiler_listener->get_available_profiles();
}

void print_options() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex_options::print_options();
    return;
}

void send (uint64_t tag, uint64_t size, uint64_t target) {
    in_apex prevent_deadlocks;
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
        //message_event_data data(tag, size, instance->get_node_id(),
        //thread_instance::get_id(), target);
        message_event_data data(tag, size, instance->get_node_id(), 0, target);
        if (_notify_listeners) {
            //read_lock_type l(instance->listener_mutex);
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_send(data);
            }
        }
    }
}

void recv (uint64_t tag, uint64_t size, uint64_t source_rank, uint64_t
    source_thread) {
    in_apex prevent_deadlocks;
    APEX_UNUSED(source_thread);
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
        //message_event_data data(tag, size, source_rank, source_thread,
        //instance->get_node_id());
        message_event_data data(tag, size, source_rank, 0,
            instance->get_node_id());
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

/* Implementation of the C API */

extern "C" {

    int apex_init(const char * thread_name, const uint64_t comm_rank,
        const uint64_t comm_size) {
        return init(thread_name, comm_rank, comm_size);
    }

#ifdef APEX_USE_STATIC_GLOBAL_CONSTRUCTOR
    void apex_init_static_void() {
        init("APEX static constructor", 0, 1);
        return;
    }

    void apex_finalize_static_void(void) {
        finalize();
    }
#endif

    int apex_init_(const uint64_t comm_rank, const uint64_t comm_size) {
        return init("FORTRAN thread", comm_rank, comm_size);
    }

    int apex_init__(const uint64_t comm_rank, const uint64_t comm_size) {
        return init("FORTRAN thread", comm_rank, comm_size);
    }

    void apex_cleanup() {
        cleanup();
    }

    const char * apex_dump(bool reset) {
        return(strdup(dump(reset).c_str()));
    }

    void apex_finalize(void) {
        finalize();
    }

    void apex_finalize_() { finalize(); }

    void apex_finalize__() { finalize(); }

    const char * apex_version() {
        return version().c_str();
    }

    apex_profiler_handle apex_start(apex_profiler_type type,
        const void * identifier) {
        APEX_ASSERT(identifier != nullptr);
        if (type == APEX_FUNCTION_ADDRESS) {
            return
            reinterpret_cast<apex_profiler_handle>(
                start((apex_function_address)identifier));
        } else if (type == APEX_NAME_STRING) {
            string tmp((const char *)identifier);
            return reinterpret_cast<apex_profiler_handle>(start(tmp));
        }
        return APEX_NULL_PROFILER_HANDLE;
    }

    void apex_start_(apex_profiler_type type, const void * identifier,
        apex_profiler_handle profiler) {
        apex_profiler_handle p = apex_start(type, identifier);
        if (profiler != nullptr) profiler = p;
    }

    apex_profiler_handle apex_resume(apex_profiler_type type,
        const void * identifier) {
        APEX_ASSERT(identifier != nullptr);
        if (type == APEX_FUNCTION_ADDRESS) {
            return reinterpret_cast<apex_profiler_handle>(
                resume((apex_function_address)identifier));
        } else if (type == APEX_NAME_STRING) {
            string tmp((const char *)identifier);
            return reinterpret_cast<apex_profiler_handle>(resume(tmp));
        }
        return APEX_NULL_PROFILER_HANDLE;
    }

    void apex_reset(apex_profiler_type type, const void * identifier) {
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

    void apex_stop(apex_profiler_handle the_profiler) {
        stop(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_stop_(apex_profiler_handle the_profiler) {
        stop(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_yield(apex_profiler_handle the_profiler) {
        yield(reinterpret_cast<profiler*>(the_profiler));
    }

    void apex_sample_value(const char * name, double value, bool threaded) {
        string tmp(name);
        sample_value(tmp, value, threaded);
    }

    void apex_new_task(apex_profiler_type type, const void * identifier,
                       unsigned long long task_id) {
        if (type == APEX_FUNCTION_ADDRESS) {
            new_task((apex_function_address)(identifier), task_id);
        } else {
            string tmp((const char *)identifier);
            new_task(tmp, task_id);
        }
    }

    apex_event_type apex_register_custom_event(const char * name) {
        string tmp(name);
        return register_custom_event(tmp);
    }

    void apex_custom_event(apex_event_type event_type,
        void * custom_data) {
        custom_event(event_type, custom_data);
    }

    void apex_register_thread(const char * name) {
        if (name) {
            string tmp(name);
            register_thread(tmp);
        } else {
            string tmp("APEX WORKER THREAD");
            register_thread(tmp);
        }
    }

    void apex_exit_thread(void) {
        exit_thread();
    }

    apex_policy_handle* apex_register_policy(const apex_event_type when,
        int (f)(apex_context const)) {
        return register_policy(when, f);
    }

    apex_policy_handle* apex_register_periodic_policy(
        unsigned long period, int (f)(apex_context const)) {
        return register_periodic_policy(period, f);
    }

    void apex_deregister_policy(apex_policy_handle * handle) {
        return deregister_policy(handle);
    }

    apex_profile* apex_get_profile(apex_profiler_type type,
        const void * identifier) {
        APEX_ASSERT(identifier != nullptr);
        if (type == APEX_FUNCTION_ADDRESS) {
            return get_profile((apex_function_address)(identifier));
        } // else {
            string tmp((const char *)identifier);
            return get_profile(tmp);
        // }
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

    void apex_recv (uint64_t tag, uint64_t size, uint64_t source_rank,
        uint64_t source_thread) {
        return recv(tag, size, source_rank, source_thread);
    }

    uint64_t apex_hardware_concurrency(void) {
        return hardware_concurrency();
    }

} // extern "C"

#ifdef APEX_HAVE_HPX
DEFINE_CONSTRUCTOR(apex_register_with_hpx);

std::shared_ptr<hpx::util::external_timer::task_wrapper> new_task_adapter(
    const std::string &name,
    const uint64_t task_id,
    const std::shared_ptr<hpx::util::external_timer::task_wrapper> parent_task)
{
    return APEX_TOP_LEVEL_PACKAGE::new_task(name, task_id,
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(parent_task));
}

std::shared_ptr<hpx::util::external_timer::task_wrapper> new_task_adapter(
    const uintptr_t address,
    const uint64_t task_id,
    const std::shared_ptr<hpx::util::external_timer::task_wrapper> parent_task)
{
    return APEX_TOP_LEVEL_PACKAGE::new_task(address, task_id,
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(parent_task));
}

std::shared_ptr<hpx::util::external_timer::task_wrapper> update_task_adapter(
    std::shared_ptr<hpx::util::external_timer::task_wrapper> wrapper,
    const std::string &timer_name) {
    return APEX_TOP_LEVEL_PACKAGE::update_task(
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(wrapper),
        timer_name);
}

std::shared_ptr<hpx::util::external_timer::task_wrapper> update_task_adapter(
    std::shared_ptr<hpx::util::external_timer::task_wrapper> wrapper,
    const uintptr_t address) {
    return APEX_TOP_LEVEL_PACKAGE::update_task(
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(wrapper),
        address);
}

void start_adapter(std::shared_ptr<hpx::util::external_timer::task_wrapper> tt_ptr) {
    APEX_TOP_LEVEL_PACKAGE::start(
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(tt_ptr));
}

void stop_adapter(std::shared_ptr<hpx::util::external_timer::task_wrapper> tt_ptr) {
    APEX_TOP_LEVEL_PACKAGE::stop(
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(tt_ptr));
}

void yield_adapter(std::shared_ptr<hpx::util::external_timer::task_wrapper> tt_ptr) {
    APEX_TOP_LEVEL_PACKAGE::yield(
        static_pointer_cast<APEX_TOP_LEVEL_PACKAGE::task_wrapper>(tt_ptr));
}

void sample_value_adapter(const std::string &name, double value) {
    APEX_TOP_LEVEL_PACKAGE::sample_value(name, value, false);
}

static void apex_register_with_hpx(void) {
    hpx::util::external_timer::registration reg;
    reg.type = hpx::util::external_timer::init_flag;
    reg.record.init = &APEX_TOP_LEVEL_PACKAGE::init;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::finalize_flag;
    reg.record.finalize = &APEX_TOP_LEVEL_PACKAGE::finalize;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::register_thread_flag;
    reg.record.register_thread = &APEX_TOP_LEVEL_PACKAGE::register_thread_hpx;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::new_task_string_flag;
    reg.record.new_task_string = &new_task_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::new_task_address_flag;
    reg.record.new_task_address = &new_task_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::sample_value_flag;
    reg.record.sample_value = &sample_value_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::send_flag;
    reg.record.send = &APEX_TOP_LEVEL_PACKAGE::send;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::recv_flag;
    reg.record.recv = &APEX_TOP_LEVEL_PACKAGE::recv;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::update_task_string_flag;
    reg.record.update_task_string = &update_task_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::update_task_address_flag;
    reg.record.update_task_address = &update_task_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::start_flag;
    reg.record.start = &start_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::stop_flag;
    reg.record.stop = &stop_adapter;
    hpx::util::external_timer::register_external_timer(reg);
    reg.type = hpx::util::external_timer::yield_flag;
    reg.record.yield = &yield_adapter;
    hpx::util::external_timer::register_external_timer(reg);
}
#endif


