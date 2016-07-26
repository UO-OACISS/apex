//  Copyright (c) 2014 University of Oregon
//

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "apex.hpp"
#include "apex_api.hpp"
#include "apex_types.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#if APEX_USE_PLUGINS
#include <dlfcn.h>
#endif
//#include <cxxabi.h> // this is for demangling strings.

#include "concurrency_handler.hpp"
#include "policy_handler.hpp"
#include "thread_instance.hpp"
#include "utils.hpp"

#ifdef APEX_HAVE_TAU
#include "tau_listener.hpp"
#define PROFILING_ON
//#define TAU_GNU
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif
#include "profiler_listener.hpp"
#ifdef APEX_DEBUG
#include "apex_error_handling.hpp"
#endif
#include "address_resolution.hpp"

#ifdef APEX_HAVE_OTF2
#include "otf2_listener.hpp"
#endif

APEX_NATIVE_TLS bool _registered = false;
APEX_NATIVE_TLS bool _exited = false;
static bool _initialized = false;

using namespace std;

namespace apex
{

// Global static pointer used to ensure a single instance of the class.
apex* apex::m_pInstance = nullptr;

std::atomic<bool> _notify_listeners(true);
std::atomic<bool> _measurement_stopped(false);
#ifdef APEX_DEBUG
std::atomic<unsigned int> _starts(0L);
std::atomic<unsigned int> _stops(0L);
std::atomic<unsigned int> _exit_stops(0L);
std::atomic<unsigned int> _resumes(0L);
std::atomic<unsigned int> _yields(0L);
#endif

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
    delete pd_reader;
#endif
    m_pInstance = nullptr;
}

void apex::set_node_id(int id)
{
    m_node_id = id;
    stringstream ss;
    ss << "locality#" << m_node_id;
    m_my_locality = string(ss.str());
    node_event_data data(id, thread_instance::get_id());
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < listeners.size() ; i++) {
            listeners[i]->on_new_node(data);
        }
    }
}

int apex::get_node_id()
{
    return m_node_id;
}

#ifdef APEX_HAVE_HPX3
static void init_hpx_runtime_ptr(void) {
    if (apex_options::disable() == true) { return; }
    apex * instance = apex::instance();
    if(instance != nullptr) {
        hpx::runtime * runtime = hpx::get_runtime_ptr();
        instance->set_hpx_runtime(runtime);
    }
}
#endif

/*
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void apex::_initialize()
{
#ifdef APEX_DEBUG
    apex_register_signal_handler();
    //apex_test_signal_handler();
#endif
    this->m_pInstance = this;
    this->m_policy_handler = nullptr;
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
#ifdef APEX_HAVE_HPX3
    this->m_hpx_runtime = nullptr;
    hpx::register_startup_function(init_hpx_runtime_ptr);
#endif
#ifdef APEX_HAVE_RCR
    energyDaemonInit();
#endif
#ifdef APEX_HAVE_MSR
    apex_init_msr();
#endif
#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau())
    {
        // before spawning any other threads, initialize TAU.
        tau_listener::initialize_tau(m_argc, m_argv);
    }
#endif
    // this is always the first listener!
    this->the_profiler_listener = new profiler_listener();
    listeners.push_back(the_profiler_listener);
#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau())
    {
        listeners.push_back(new tau_listener());
    }
#endif
#ifdef APEX_HAVE_OTF2
    if (apex_options::use_otf2())
    {
        listeners.push_back(new otf2_listener());
    }
#endif
    startup_throttling();
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
    pd_reader = new proc_data_reader();
#endif
    this->resize_state(1);
    this->set_state(0, APEX_BUSY);
}

/*  
    This function is called to create an instance of the class.
    Calling the constructor publicly is not allowed. The constructor
    is private and is only called by this Instance function.
*/
apex* apex::instance()
{
    // Only allow one instance of class to be generated.
    if (m_pInstance == nullptr && !_measurement_stopped)
    {
        m_pInstance = new apex;
    }
    else if (_measurement_stopped) return nullptr;
    return m_pInstance;
}

// special case - for cleanup only!
apex* apex::__instance()
{
    return m_pInstance;
}

apex* apex::instance(int argc, char**argv)
{
    // Only allow one instance of class to be generated.
    if (m_pInstance == nullptr && !_measurement_stopped)
    {
        m_pInstance = new apex(argc, argv);
    }
    else if (_measurement_stopped) return nullptr;
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

int initialize_worker_thread_for_TAU(void) {
#ifdef APEX_HAVE_TAU
  if (apex_options::use_tau())
  {
    //if (thread_instance::get_id() > 0) {
      TAU_REGISTER_THREAD();
    //}
    Tau_create_top_level_timer_if_necessary();
  }
#endif
  return 0;
}

void init(const char * thread_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // protect against multiple initializations
    if (_registered || _initialized) { return; }
    _registered = true;
    _initialized = true;
    int argc = 1;
    const char *dummy = "APEX Application";
    char* argv[1];
    argv[0] = const_cast<char*>(dummy);
    apex* instance = apex::instance(argc, argv); // get/create the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    init_plugins();
    startup_event_data data(argc, argv);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(data);
        }
    }
#if HAVE_TAU_disabled
    // start top-level timers for threads
    if (thread_name) {
      start(thread_name);
    } else {
      start("APEX MAIN THREAD");
    }
#else 
    APEX_UNUSED(thread_name);
#endif
    if (apex_options::use_screen_output() && instance->get_node_id() == 0) {
	  std::cout << version() << std::endl;
      apex_options::print_options();
	}
    if (apex_options::throttle_energy() && apex_options::throttle_concurrency() ) {
      setup_power_cap_throttling();
    }
}

void init(int argc, char** argv, const char * thread_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // protect against multiple initializations
    if (_registered || _initialized) { return; }
    _registered = true;
    _initialized = true;
    apex* instance = apex::instance(argc, argv); // get/create the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    init_plugins();
    startup_event_data data(argc, argv);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_startup(data);
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
    if (apex_options::use_screen_output() && instance->get_node_id() == 0) {
	  std::cout << version() << std::endl;
      apex_options::print_options();
	}
    if (apex_options::throttle_energy() && apex_options::throttle_concurrency() ) {
      setup_power_cap_throttling();
    }
}

string& version() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { static string tmp("disabled"); return tmp; }
    apex* instance = apex::instance(); // get the Apex static instance
    return instance->version_string;
}

profiler* start(task_identifier * id) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
#ifdef APEX_DEBUG
    _starts++;
#endif
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return profiler::get_disabled_profiler(); }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return nullptr; // protect against calls after finalization
    if (_notify_listeners) {
        bool success = true;
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            success = instance->listeners[i]->on_start(id);
            if (!success) {
                return profiler::get_disabled_profiler();
            }
        }
    }
    return thread_instance::instance().get_current_profiler();
}

profiler* start(const std::string &timer_name)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    if (starts_with(timer_name, string("apex_internal"))) {
        return profiler::get_disabled_profiler(); // don't process our own events - queue scrubbing tasks.
    }
#ifdef APEX_HAVE_HPX3
    if (starts_with(timer_name, string("shutdown_all"))) {
        return profiler::get_disabled_profiler();
    }
#endif
    task_identifier * id = new task_identifier(timer_name);
    return start(id);
}

profiler* start(apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
    task_identifier * id = new task_identifier(function_address);
    return start(id);
}

profiler* resume(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
#ifdef APEX_DEBUG
    _resumes++;
#endif
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return profiler::get_disabled_profiler(); }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return nullptr; // protect against calls after finalization
    if (starts_with(timer_name, string("apex_internal"))) {
        return profiler::get_disabled_profiler(); // don't process our own events
    }
    if (_notify_listeners) {
        task_identifier * id = new task_identifier(timer_name);
        try {
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_resume(id);
            }
        } catch (disabled_profiler_exception e) { return profiler::get_disabled_profiler(); }
    }
    return thread_instance::instance().get_current_profiler();
}

profiler* resume(apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
#ifdef APEX_DEBUG
    _resumes++;
#endif
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return profiler::get_disabled_profiler(); }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return nullptr; // protect against calls after finalization
    if (_notify_listeners) {
        task_identifier * id = new task_identifier(function_address);
        try {
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_resume(id);
            }
        } catch (disabled_profiler_exception e) { return profiler::get_disabled_profiler(); }
    }
#ifdef APEX_DEBUG
/*
    if (instance->get_node_id() == 0) { 
        printf("%lu Resume: %s %p\n", thread_instance::get_id(), lookup_address((uintptr_t)function_address, false)->c_str(), thread_instance::instance().get_current_profiler());
        fflush(stdout); 
    }
*/
#endif
    return thread_instance::instance().get_current_profiler();
}

void reset(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    task_identifier * id = new task_identifier(timer_name);
    instance->the_profiler_listener->reset(id);
}

void reset(apex_function_address function_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
	if (function_address == APEX_NULL_FUNCTION_ADDRESS) {
        instance->the_profiler_listener->reset_all();
	} else {
    	task_identifier * id = new task_identifier(function_address);
    	instance->the_profiler_listener->reset(id);
    }
}

void set_state(apex_thread_state state) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    instance->set_state(thread_instance::get_id(), state);
}

void stop(profiler* the_profiler) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_DEBUG
    _stops++;
#endif
    if (the_profiler == profiler::get_disabled_profiler()) return; // profiler was throttled.

    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    if (the_profiler == nullptr || the_profiler->stopped) return;
    std::shared_ptr<profiler> p{the_profiler};
    /*
    std::shared_ptr<profiler> p;
    // A null profiler is OK, it means the application didn't store it. We have it.
    if (the_profiler == nullptr) {
        p = std::make_shared<profiler>(thread_instance::instance().pop_current_profiler());
    } else {
        p = std::make_shared<profiler>(thread_instance::instance().pop_current_profiler(the_profiler));
    }
    if (p == nullptr) return;
    */
#ifdef APEX_DEBUG
    /*
    if (instance->get_node_id() == 0) { 
        printf("%lu Stop:  %s %p\n", thread_instance::get_id(), lookup_address((uintptr_t)p->action_address, false)->c_str(), the_profiler);
        fflush(stdout); 
    }
    */
#endif
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_stop(p);
        }
    }
}

void yield(profiler* the_profiler)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_DEBUG
    _yields++;
#endif
    if (the_profiler == profiler::get_disabled_profiler()) return; // profiler was throttled.

    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    if (the_profiler == nullptr || the_profiler->stopped) return;
    std::shared_ptr<profiler> p{the_profiler};
    /*
    std::shared_ptr<profiler> p;
    if (the_profiler == nullptr) {
        p = std::make_shared<profiler>(thread_instance::instance().pop_current_profiler());
    } else {
        p = std::make_shared<profiler>(thread_instance::instance().pop_current_profiler(the_profiler));
    }
    if (p == nullptr) return;
    */
#ifdef APEX_DEBUG
    /*
    if (instance->get_node_id() == 0) { 
        printf("%lu Yield:  %s\n", thread_instance::get_id(), lookup_address((uintptr_t)p->action_address, false)->c_str());
        fflush(stdout); 
    }
    */
#endif
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_yield(p);
        }
    }
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
    sample_value_event_data* data = nullptr;
    if (name.find(instance->m_my_locality) != name.npos)
    {
        if (name.find("worker-thread") != name.npos)
        {
            string tmp_name = string(name.c_str());
            // tokenize by / character
            char* token = strtok(const_cast<char*>(tmp_name.c_str()), "/");
            while (token!=nullptr) {
              if (strstr(token, "worker-thread")==NULL)
              {
                break;
              }
              token = strtok(NULL, "/");
            }
            int tid = 0;
            if (token != nullptr) {
              // strip the trailing close bracket
              token = strtok(token, "}");
              tid = thread_instance::map_name_to_id(token);
            }
            if (tid != -1)
            {
                data = new sample_value_event_data(tid, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, tid);
            }
            else
            {
                data = new sample_value_event_data(0, name, value);
                //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
            }
        }
        else
        {
            data = new sample_value_event_data(0, name, value);
            //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
        }
    }
    else
    {
        // what if it doesn't?
        data = new sample_value_event_data(0, name, value);
    }
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_sample_value(*data);
        }
    }
    delete(data);
}

void new_task(task_identifier * task_id, void * data) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    new_task_event_data * event_data = new new_task_event_data(task_id, data);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_task(*event_data);
        }
    }
    delete(event_data);
}

void new_task(const std::string &timer_name, void * task_id)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    task_identifier * id = new task_identifier(timer_name);
    new_task(id, task_id);
}

void new_task(apex_function_address function_address, void * task_id) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    task_identifier * id = new task_identifier(function_address);
    new_task(id, task_id);
}

void new_dependency(task_identifier * src, task_identifier * dest) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    new_dependency_event_data * event_data = new new_dependency_event_data(src, dest);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_dependency(*event_data);
        }
    }
    delete(event_data);
}

void satisfy_dependency(task_identifier * src, task_identifier * dest) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // if APEX is suspended, do nothing.
    if (apex_options::suspend() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    satisfy_dependency_event_data * event_data = new satisfy_dependency_event_data(src, dest);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_satisfy_dependency(*event_data);
        }
    }
    delete(event_data);
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
    std::unique_lock<std::mutex> l(instance->custom_event_mutex);
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
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_custom_event(data);
        }
    }
}


void set_node_id(int id)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance();
    if (!instance || _exited) return; // protect against calls after finalization
    instance->set_node_id(id);
}

#ifdef APEX_HAVE_HPX3
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
            dlclose(plugin_handle);
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
}

void track_power(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_HAVE_TAU
    TAU_TRACK_POWER();
#endif
}

void track_power_here(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_HAVE_TAU
    TAU_TRACK_POWER_HERE();
#endif
}

void enable_tracking_power(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_HAVE_TAU
    TAU_ENABLE_TRACKING_POWER();
#endif
}

void disable_tracking_power(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_HAVE_TAU
    TAU_DISABLE_TRACKING_POWER();
#endif
}

void set_interrupt_interval(int seconds)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
#ifdef APEX_HAVE_TAU
    TAU_SET_INTERRUPT_INTERVAL(seconds);
#else 
    APEX_UNUSED(seconds);
#endif
}

void finalize()
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    shutdown_throttling(); // if not done already
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance) return; // protect against calls after finalization
    finalize_plugins();
    exit_thread();
#if APEX_HAVE_PROC
    instance->pd_reader->stop_reading();
#endif
#if APEX_HAVE_MSR
    apex_finalize_msr();
#endif
    if (!_measurement_stopped)
    {
        _measurement_stopped = true;
#ifdef APEX_DEBUG
        std::cout << instance->get_node_id() << " Starts  : " << _starts  << std::endl;
        std::cout << instance->get_node_id() << " Resumes : " << _resumes << std::endl;
        std::cout << instance->get_node_id() << " Yields  : " << _yields  << std::endl;
        std::cout << instance->get_node_id() << " Stops   : " << _stops   << std::endl;
        std::cout << instance->get_node_id() << " Exit Stops   : " << _exit_stops   << std::endl;
        unsigned int ins = _starts + _resumes;
        unsigned int outs = _yields + _stops + _exit_stops;
        if (ins != outs) {
            std::cout << std::endl;
            std::cout << " ------->>> ERROR! missing ";
            if (ins > outs) {
              std::cout << (ins - outs) << " stops. <<<-------" << std::endl;
            } else {
              std::cout << (outs - ins) << " starts. <<<-------" << std::endl;
            }
            std::cout << std::endl;
            //assert(ins == outs);
        }
#endif
        stringstream ss;
        ss << instance->get_node_id();
        shutdown_event_data data(instance->get_node_id(), thread_instance::get_id());
        _notify_listeners = false;
        //if (_notify_listeners) {
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_shutdown(data);
            }
        //}
    }
}

void cleanup(void) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::__instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against multiple calls
    if (!_measurement_stopped) {
        finalize();
    }
    delete(instance);
}

void register_thread(const std::string &name)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    if (_registered) return; // protect against multiple registrations on the same thread
    thread_instance::set_name(name);
    instance->resize_state(thread_instance::get_id());
    instance->set_state(thread_instance::get_id(), APEX_BUSY);
    new_thread_event_data data(name);
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_new_thread(data);
        }
    }
#ifdef APEX_HAVE_TAU_disabled
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

void exit_thread(void)
{
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex* instance = apex::instance(); // get the Apex static instance
    if (!instance || _exited) return; // protect against calls after finalization
    _exited = true;
    /*
    // pop any remaining timers, and stop them
    std::shared_ptr<profiler> p;
    while(true && !thread_instance::instance().profiler_stack_empty()) {
        p = std::make_shared<profiler>(thread_instance::instance().pop_current_profiler());
        if (p == nullptr) { break; }
#ifdef APEX_DEBUG
        _exit_stops++;
#endif
        if (_notify_listeners) {
            for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
                instance->listeners[i]->on_stop(p);
            }
        }
    }
    */
    event_data data;
    if (_notify_listeners) {
        for (unsigned int i = 0 ; i < instance->listeners.size() ; i++) {
            instance->listeners[i]->on_exit_thread(data);
        }
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
    return handle;
}

void deregister_policy(apex_policy_handle * handle) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    // disable processing of policy for now
    //_notify_listeners = false;
    policy_handler * handler = apex::instance()->get_policy_handler();
    if(handler != nullptr) {
        handler->deregister_policy(handle);
    }
    //_notify_listeners = true;
    delete(handle);
}

apex_profile* get_profile(apex_function_address action_address) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
	task_identifier id = task_identifier(action_address);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

apex_profile* get_profile(const std::string &timer_name) {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return nullptr; }
	task_identifier id = task_identifier(timer_name);
    profile * tmp = apex::__instance()->the_profiler_listener->get_profile(id);
    if (tmp != nullptr)
        return tmp->get_profile();
    return nullptr;
}

/*
std::vector<std::string> get_available_profiles() {
    return apex::__instance()->the_profiler_listener->get_available_profiles();
}
*/

void print_options() {
    // if APEX is disabled, do nothing.
    if (apex_options::disable() == true) { return; }
    apex_options::print_options();
    return;
}



} // apex namespace

using namespace apex;

extern "C" {

    void apex_init(const char * thread_name)
    {
        init(thread_name);
    }

    void apex_init_(void) { init("FORTRAN thread"); }

    void apex_init__(void) { init("FORTRAN thread"); }

    void apex_init_args(int argc, char** argv, const char * thread_name)
    {
        init(argc, argv, thread_name);
    }

    void apex_cleanup()
    {
        cleanup();
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
                       void * task_id) {
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

    void apex_set_node_id(int id)
    {
        set_node_id(id);
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

} // extern "C"


