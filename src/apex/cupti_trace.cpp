/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 * Adapted from CUDA/CUPTI example from NVIDIA:
 * Copyright 2011-2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 * using asynchronous handling of activity buffers.
 *
 */

#include <stdio.h>
#include <stack>
#include <unordered_map>
#include <map>
#include <sstream>
#include <mutex>
#include <atomic>
#include "apex.hpp"
#include "profiler.hpp"
#include "thread_instance.hpp"
#include "apex_options.hpp"
#if defined(APEX_WITH_PERFETTO)
#include "perfetto_listener.hpp"
#endif
#include "trace_event_listener.hpp"
#include "apex_nvml.hpp"
#ifdef APEX_HAVE_OTF2
#include "otf2_listener.hpp"
#endif
#include "async_thread_node.hpp"
#include "memory_wrapper.hpp"

#include <cuda_runtime_api.h>
#include <cupti.h>
#include <nvToolsExt.h>
#include <nvToolsExtSync.h>
#include <generated_nvtx_meta.h>
#include <dlfcn.h>

/* Fun!  CUPTI doesn't do callbacks for end or push events.  Wheeeeee
 * So, what we'll do is wrap the functions instead of having callbacks. */
#define APEX_BROKEN_CUPTI_NVTX_PUSH_POP 1

#define CUPTI_CALL(call)                                                \
    do {                                                                  \
        CUptiResult _status = call;                                         \
        if (_status != CUPTI_SUCCESS) {                                     \
            const char *errstr;                                               \
            cuptiGetResultString(_status, &errstr);                           \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                       \
            exit(-1);                                                     \
        }                                                                   \
    } while (0)

#define CUDA_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)


#define BUF_SIZE (1024 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
  char const * err = dlerror(); \
  if (err) { \
    printf("Error getting %s handle: %s\n", name, err); \
    fflush(stdout); \
    exit(1); \
  } \
}

static
void * get_system_function_handle(char const * name, void * caller)
{
  void * handle;

  // Reset error pointer
  RESET_DLERROR();

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

  // Detect errors
  CHECK_DLERROR();

  // Prevent recursion if more than one wrapping approach has been loaded.
  // This happens because we support wrapping pthreads three ways at once:
  // #defines in Profiler.h, -Wl,-wrap on the link line, and LD_PRELOAD.
  if (handle == caller) {
    RESET_DLERROR();
    void * syms = dlopen(NULL, RTLD_NOW);
    CHECK_DLERROR();
    do {
      RESET_DLERROR();
      handle = dlsym(syms, name);
      CHECK_DLERROR();
    } while (handle == caller);
  }

  return handle;
}

std::map<nvtxDomainHandle_t, std::string>& get_domain_map() {
    static std::map<nvtxDomainHandle_t, std::string> the_map;
    return the_map;
}

std::map<nvtxRangeId_t, apex::profiler*>& get_range_map() {
    static std::map<nvtxRangeId_t, apex::profiler*> the_map;
    return the_map;
}

std::stack<std::shared_ptr<apex::task_wrapper> >& get_range_stack() {
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > the_stack;
    return the_stack;
}

/* Wrap some NVTX functions
 *
 * Because CUPTI doesn't give us a way to map domain/range names to their
 * ids, we have to intercept the function calls and manage them ourselves.
 */

// forward declare a useful function, defined below
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib);

// Make sure we have C binding on these things
extern "C" {

// Some typedefs for functions that we are wrapping
typedef nvtxDomainHandle_t (*nvtxDomainCreateA_p)(const char * name);
typedef nvtxRangeId_t (*nvtxDomainRangeStartEx_p)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
typedef nvtxRangeId_t (*nvtxRangeStartEx_p)(const nvtxEventAttributes_t *eventAttrib);
typedef nvtxRangeId_t (*nvtxRangeStartA_p)(const char * message);
typedef nvtxRangeId_t (*nvtxRangeStartW_p)(const wchar_t * message);
#ifdef APEX_BROKEN_CUPTI_NVTX_PUSH_POP
typedef void (*nvtxDomainRangeEnd_p)(nvtxDomainHandle_t domain, nvtxRangeId_t id);
typedef void (*nvtxRangeEnd_p)(nvtxRangeId_t id);
typedef int (*nvtxDomainRangePushEx_p)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
typedef int (*nvtxRangePushEx_p)(const nvtxEventAttributes_t *eventAttrib);
typedef int (*nvtxRangePushA_p)(const char * message);
typedef int (*nvtxRangePushW_p)(const wchar_t * message);
typedef int (*nvtxDomainRangePop_p)(nvtxDomainHandle_t domain);
typedef int (*nvtxRangePop_p)(void);
#endif

/* Define the wrapper for nvtxDomainCreateA */
nvtxDomainHandle_t apex_nvtxDomainCreateA_wrapper(
    nvtxDomainCreateA_p nvtxDomainCreateA_call, const char * name) {
    auto handle = nvtxDomainCreateA_call(name);
    std::string tmp{name};
    get_domain_map().insert(std::pair<nvtxDomainHandle_t, std::string>(handle, tmp));
    return handle;
}

/* Define the interceptor for nvtxDomainCreateA */
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateA (const char * name) {
    static nvtxDomainCreateA_p _nvtxDomainCreateA =
        (nvtxDomainCreateA_p)(get_system_function_handle("nvtxDomainCreateA", (void*)(nvtxDomainCreateA)));
    return apex_nvtxDomainCreateA_wrapper(_nvtxDomainCreateA, name);
}

/* Define the common wrapper for a range timer */
void apex_nvtxRangeStart (nvtxRangeId_t id, const std::string name) {
    auto p = apex::start(name);
    get_range_map().insert(std::pair<nvtxRangeId_t, apex::profiler*>(id, p));
}

void apex_nvtxRangePush (const std::string name) {
    auto timer = apex::new_task(name);
    apex::start(timer);
    get_range_stack().push(timer);
}

/* Define the wrapper for nvtxRangeStartA */
nvtxRangeId_t apex_nvtxRangeStartA_wrapper (
    nvtxRangeStartA_p nvtxRangeStartA_call, const char * message) {
    auto handle = nvtxRangeStartA_call(message);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::string tmp{message};
        apex_nvtxRangeStart(handle, tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxRangeStartA */
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartA (const char * message) {
    static nvtxRangeStartA_p _nvtxRangeStartA =
        (nvtxRangeStartA_p)(get_system_function_handle("nvtxRangeStartA", (void*)(nvtxRangeStartA)));
    return apex_nvtxRangeStartA_wrapper(_nvtxRangeStartA, message);
}

#ifdef APEX_BROKEN_CUPTI_NVTX_PUSH_POP
/* Define the wrapper for nvtxRangePushA */
int apex_nvtxRangePushA_wrapper (
    nvtxRangePushA_p nvtxRangePushA_call, const char * message) {
    auto handle = nvtxRangePushA_call(message);
    std::string tmp{message};
    apex_nvtxRangePush(tmp);
    return handle;
}

/* Define the interceptor for nvtxRangePushA */
NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char * message) {
    static nvtxRangePushA_p _nvtxRangePushA =
        (nvtxRangePushA_p)(get_system_function_handle("nvtxRangePushA", (void*)(nvtxRangePushA)));
    return apex_nvtxRangePushA_wrapper(_nvtxRangePushA, message);
}

/* Define the wrapper for nvtxRangeStartW */
nvtxRangeId_t apex_nvtxRangeStartW_wrapper (
    nvtxRangeStartW_p nvtxRangeStartW_call, const wchar_t * message) {
    auto handle = nvtxRangeStartW_call(message);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::wstring wtmp(message);
        std::string tmp = std::string(wtmp.begin(), wtmp.end());
        apex_nvtxRangeStart(handle, tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxRangeStartW */
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartW (const wchar_t * message) {
    static nvtxRangeStartW_p _nvtxRangeStartW =
        (nvtxRangeStartW_p)(get_system_function_handle("nvtxRangeStartW", (void*)(nvtxRangeStartW)));
    return apex_nvtxRangeStartW_wrapper(_nvtxRangeStartW, message);
}

/* Define the wrapper for nvtxRangePushW */
int apex_nvtxRangePushW_wrapper (
    nvtxRangePushW_p nvtxRangePushW_call, const wchar_t * message) {
    auto handle = nvtxRangePushW_call(message);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::wstring wtmp(message);
        std::string tmp = std::string(wtmp.begin(), wtmp.end());
        apex_nvtxRangePush(tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxRangePushW */
NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t * message) {
    static nvtxRangePushW_p _nvtxRangePushW =
        (nvtxRangePushW_p)(get_system_function_handle("nvtxRangePushW", (void*)(nvtxRangePushW)));
    return apex_nvtxRangePushW_wrapper(_nvtxRangePushW, message);
}

void apex_nvtxRangePop (void) {
    if (!get_range_stack().empty()) {
        auto timer = get_range_stack().top();
        apex::stop(timer);
        get_range_stack().pop();
    }
}

/* Define the wrapper for nvtxRangePop */
int apex_nvtxRangePop_wrapper (nvtxRangePop_p nvtxRangePop_call) {
    auto handle = nvtxRangePop_call();
    apex_nvtxRangePop();
    return handle;
}

/* Define the interceptor for nvtxRangePop */
NVTX_DECLSPEC int NVTX_API nvtxRangePop (void) {
    static nvtxRangePop_p _nvtxRangePop =
        (nvtxRangePop_p)(get_system_function_handle("nvtxRangePop", (void*)(nvtxRangePop)));
    return apex_nvtxRangePop_wrapper(_nvtxRangePop);
}

/* Define the wrapper for nvtxDomainRangePop */
int apex_nvtxDomainRangePop_wrapper (nvtxDomainRangePop_p nvtxDomainRangePop_call,
    nvtxDomainHandle_t domain) {
    auto handle = nvtxDomainRangePop_call(domain);
    apex_nvtxRangePop();
    return handle;
}

/* Define the interceptor for nvtxDomainRangePop */
NVTX_DECLSPEC int NVTX_API nvtxDomainRangePop (nvtxDomainHandle_t domain) {
    static nvtxDomainRangePop_p _nvtxDomainRangePop =
        (nvtxDomainRangePop_p)(get_system_function_handle("nvtxDomainRangePop", (void*)(nvtxDomainRangePop)));
    return apex_nvtxDomainRangePop_wrapper(_nvtxDomainRangePop, domain);
}

void apex_nvtxRangeEnd(nvtxRangeId_t id) {
    /* Range start/end is too risky for OTF2 */
    if (apex::apex_options::use_otf2()) { return; }
    apex::stop(get_range_map()[id]);
    get_range_map().erase(id);
}

/* Define the wrapper for nvtxRangeEnd */
void apex_nvtxRangeEnd_wrapper (
    nvtxRangeEnd_p nvtxRangeEnd_call,
    const nvtxRangeId_t id) {
    nvtxRangeEnd_call(id);
    apex_nvtxRangeEnd(id);
    return;
}

/* Define the interceptor for nvtxRangeEnd */
NVTX_DECLSPEC void NVTX_API nvtxRangeEnd (nvtxRangeId_t id) {
    static nvtxRangeEnd_p _nvtxRangeEnd =
        (nvtxRangeEnd_p)(get_system_function_handle("nvtxRangeEnd", (void*)(nvtxRangeEnd)));
    return apex_nvtxRangeEnd_wrapper(_nvtxRangeEnd, id);
}

/* Define the wrapper for nvtxDomainRangeEnd */
void apex_nvtxDomainRangeEnd_wrapper (
    nvtxDomainRangeEnd_p nvtxDomainRangeEnd_call,
    nvtxDomainHandle_t domain,
    const nvtxRangeId_t id) {
    nvtxDomainRangeEnd_call(domain, id);
    apex_nvtxRangeEnd(id);
    return;
}

/* Define the interceptor for nvtxDomainRangeEnd */
NVTX_DECLSPEC void NVTX_API nvtxDomainRangeEnd (nvtxDomainHandle_t domain, nvtxRangeId_t id) {
    static nvtxDomainRangeEnd_p _nvtxDomainRangeEnd =
        (nvtxDomainRangeEnd_p)(get_system_function_handle("nvtxDomainRangeEnd", (void*)(nvtxDomainRangeEnd)));
    return apex_nvtxDomainRangeEnd_wrapper(_nvtxDomainRangeEnd, domain, id);
}

#endif

/* Define the wrapper for nvtxRangeStartEx */
nvtxRangeId_t apex_nvtxRangeStartEx_wrapper (
    nvtxRangeStartEx_p nvtxRangeStartEx_call,
    const nvtxEventAttributes_t *eventAttrib) {
    auto handle = nvtxRangeStartEx_call(eventAttrib);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::string tmp{get_nvtx_message(eventAttrib)};
        apex_nvtxRangeStart(handle, tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxRangeStartEx */
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartEx (const nvtxEventAttributes_t *eventAttrib) {
    static nvtxRangeStartEx_p _nvtxRangeStartEx =
        (nvtxRangeStartEx_p)(get_system_function_handle("nvtxRangeStartEx", (void*)(nvtxRangeStartEx)));
    return apex_nvtxRangeStartEx_wrapper(_nvtxRangeStartEx, eventAttrib);
}

/* Define the wrapper for nvtxDomainRangeStartEx */
nvtxRangeId_t apex_nvtxDomainRangeStartEx_wrapper (
    nvtxDomainRangeStartEx_p nvtxDomainRangeStartEx_call,
    nvtxDomainHandle_t domain,
    const nvtxEventAttributes_t *eventAttrib) {
    auto handle = nvtxDomainRangeStartEx_call(domain, eventAttrib);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::string tmp;
        if (domain != NULL) {
            std::string domain_name(get_domain_map()[domain]);
            std::stringstream ss;
            ss << domain_name << ": " << get_nvtx_message(eventAttrib);
            tmp = ss.str();
        } else {
            tmp = get_nvtx_message(eventAttrib);
        }
        apex_nvtxRangeStart(handle, tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxDomainRangeStartEx */
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxDomainRangeStartEx (nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) {
    static nvtxDomainRangeStartEx_p _nvtxDomainRangeStartEx =
        (nvtxDomainRangeStartEx_p)(get_system_function_handle("nvtxDomainRangeStartEx", (void*)(nvtxDomainRangeStartEx)));
    return apex_nvtxDomainRangeStartEx_wrapper(_nvtxDomainRangeStartEx, domain, eventAttrib);
}

/* Define the wrapper for nvtxRangePushEx */
int apex_nvtxRangePushEx_wrapper (
    nvtxRangePushEx_p nvtxRangePushEx_call,
    const nvtxEventAttributes_t *eventAttrib) {
    auto handle = nvtxRangePushEx_call(eventAttrib);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::string tmp{get_nvtx_message(eventAttrib)};
        apex_nvtxRangePush(tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxRangePushEx */
NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib) {
    static nvtxRangePushEx_p _nvtxRangePushEx =
        (nvtxRangePushEx_p)(get_system_function_handle("nvtxRangePushEx", (void*)(nvtxRangePushEx)));
    return apex_nvtxRangePushEx_wrapper(_nvtxRangePushEx, eventAttrib);
}

/* Define the wrapper for nvtxDomainRangePushEx */
int apex_nvtxDomainRangePushEx_wrapper (
    nvtxDomainRangePushEx_p nvtxDomainRangePushEx_call,
    nvtxDomainHandle_t domain,
    const nvtxEventAttributes_t *eventAttrib) {
    auto handle = nvtxDomainRangePushEx_call(domain, eventAttrib);
    /* Range start/end is too risky for OTF2 */
    if (!apex::apex_options::use_otf2()) {
        std::string tmp;
        if (domain != NULL) {
            std::string domain_name(get_domain_map()[domain]);
            std::stringstream ss;
            ss << domain_name << ": " << get_nvtx_message(eventAttrib);
            tmp = ss.str();
        } else {
            tmp = get_nvtx_message(eventAttrib);
        }
        apex_nvtxRangePush(tmp);
    }
    return handle;
}

/* Define the interceptor for nvtxDomainRangePushEx */
NVTX_DECLSPEC int NVTX_API nvtxDomainRangePushEx (nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) {
    static nvtxDomainRangePushEx_p _nvtxDomainRangePushEx =
        (nvtxDomainRangePushEx_p)(get_system_function_handle("nvtxDomainRangePushEx", (void*)(nvtxDomainRangePushEx)));
    return apex_nvtxDomainRangePushEx_wrapper(_nvtxDomainRangePushEx, domain, eventAttrib);
}

} /* extern "C" */

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestampGPU{0};
static uint64_t startTimestampCPU{0};
static int64_t deltaTimestamp{0};

/* The callback subscriber */
CUpti_SubscriberHandle subscriber;

/* Global flag to indicate we are working */
static std::atomic<bool> allGood{false};

/* The buffer count */
std::atomic<uint64_t> num_buffers{0};
std::atomic<uint64_t> num_buffers_processed{0};
bool flushing{false};

/* The map that holds correlation IDs and matches them to GUIDs */
std::unordered_map<uint32_t, std::shared_ptr<apex::task_wrapper>> correlation_map;
/* The map that holds context IDs and matches them to device IDs */
std::unordered_map<uint32_t, uint32_t> context_map;
std::unordered_map<uint32_t, apex::async_event_data> correlation_kernel_data_map;
std::mutex map_mutex;

/* Make sure APEX knows about this thread */
bool& get_registered(void) {
    static APEX_NATIVE_TLS bool registered{false};
    return registered;
}

bool register_myself(bool isWorker = true) {
    bool& registered = get_registered();
    if (!registered && !isWorker) {
        // make sure APEX knows this is not a worker thread
        apex::thread_instance::instance(false);
        /* make sure the profiler_listener has a queue that this
         * thread can push sampled values to */
        apex::apex::async_thread_setup();
        registered = true;
    } else if (!registered && isWorker) {
        apex::register_thread("APEX CUPTI support");
    }
    return registered;
}

void store_profiler_data(const std::string &name, uint32_t correlationId,
        uint64_t start, uint64_t end, apex::cuda_thread_node &node,
        std::string category, bool reverseFlow = false, bool otf2_trace = true) {
    apex::in_apex prevent_deadlocks;
    // Get the singleton APEX instance
    static apex::apex* instance = apex::apex::instance();
    // get the parent GUID, then erase the correlation from the map
    std::shared_ptr<apex::task_wrapper> parent = nullptr;
    apex::async_event_data as_data;
    if (correlationId > 0) {
        map_mutex.lock();
        parent = correlation_map[correlationId];
        as_data = correlation_kernel_data_map[correlationId];
        correlation_map.erase(correlationId);
        correlation_kernel_data_map.erase(correlationId);
        map_mutex.unlock();
    }
    // Build the name
    std::stringstream ss;
    ss << "GPU: " << std::string(name);
    std::string tmp{ss.str()};
    // create a task_wrapper, as a GPU child of the parent on the CPU side
    auto tt = apex::new_task(tmp, UINT64_MAX, parent);
    // create an APEX profiler to store this data - we can't start
    // then stop because we have timestamps already.
    auto prof = std::make_shared<apex::profiler>(tt);
    prof->set_start(start + deltaTimestamp);
    prof->set_end(end + deltaTimestamp);
    // important!  Otherwise we might get the wrong end timestamp.
    prof->stopped = true;
    // fake out the profiler_listener
    instance->the_profiler_listener->push_profiler_public(prof);
    // Handle tracing, if necessary
#if defined(APEX_WITH_PERFETTO)
    if (apex::apex_options::use_perfetto()) {
        apex::perfetto_listener * tel =
            (apex::perfetto_listener*)instance->the_perfetto_listener;
        as_data.cat = category;
        as_data.reverse_flow = reverseFlow;
        tel->on_async_event(node, prof, as_data);
    }
#endif
    if (apex::apex_options::use_trace_event()) {
        apex::trace_event_listener * tel =
            (apex::trace_event_listener*)instance->the_trace_event_listener;
        as_data.cat = category;
        as_data.reverse_flow = reverseFlow;
        tel->on_async_event(node, prof, as_data);
    }
#ifdef APEX_HAVE_OTF2
    if (apex::apex_options::use_otf2() && otf2_trace) {
        apex::otf2_listener * tol =
            (apex::otf2_listener*)instance->the_otf2_listener;
        tol->on_async_event(node, prof);
    }
#else
    APEX_UNUSED(otf2_trace);
#endif
    // have the listeners handle the end of this task
    instance->complete_task(tt);
}

/* Handle counters from synchronous callbacks */
void store_sync_counter_data(const char * name, const std::string& context,
    double value, bool force = false, bool threaded = true) {
    if (name == nullptr) {
        apex::sample_value(context, value, true);
    } else {
        std::stringstream ss;
        ss << name;
        if (apex::apex_options::use_cuda_kernel_details() || force) {
            ss << ": " << context;
        }
        apex::sample_value(ss.str(), value, threaded);
    }
}

/* Handle counters from asynchronous activity */
void store_counter_data(const char * name, const std::string& ctx,
    uint64_t end, double value, apex::cuda_thread_node &node, bool force = false) {
    apex::in_apex prevent_deadlocks;
    std::stringstream ss;
    if (name == nullptr) {
        ss << ctx;
    } else {
        ss << name;
        if (apex::apex_options::use_cuda_kernel_details() || force) {
            ss << ": " << ctx;
        }
    }
    std::string tmp{ss.str()};
    auto task_id = apex::task_identifier::get_task_id(tmp);
    std::shared_ptr<apex::profiler> prof =
        std::make_shared<apex::profiler>(task_id, value);
    prof->is_counter = true;
    prof->set_end(end + deltaTimestamp);
    // Get the singleton APEX instance
    static apex::apex* instance = apex::apex::instance();
    // fake out the profiler_listener
    instance->the_profiler_listener->push_profiler_public(prof);
    // Handle tracing, if necessary
#if defined(APEX_WITH_PERFETTO)
    if (apex::apex_options::use_perfetto()) {
        apex::perfetto_listener * tel =
            (apex::perfetto_listener*)instance->the_perfetto_listener;
        tel->on_async_metric(node, prof);
    }
#endif
    if (apex::apex_options::use_trace_event()) {
        apex::trace_event_listener * tel =
            (apex::trace_event_listener*)instance->the_trace_event_listener;
        tel->on_async_metric(node, prof);
    }
#ifdef APEX_HAVE_OTF2
    if (apex::apex_options::use_otf2()) {
        apex::otf2_listener * tol =
            (apex::otf2_listener*)instance->the_otf2_listener;
        tol->on_async_metric(node, prof);
    }
#endif
}

void store_counter_data(const char * name, const std::string& ctx,
    uint64_t end, int32_t value, apex::cuda_thread_node &node, bool force = false) {
    store_counter_data(name, ctx, end, (double)(value), node, force);
}

void store_counter_data(const char * name, const std::string& ctx,
    uint64_t end, uint32_t value, apex::cuda_thread_node &node, bool force = false) {
    store_counter_data(name, ctx, end, (double)(value), node, force);
}

void store_counter_data(const char * name, const std::string& ctx,
    uint64_t end, uint64_t value, apex::cuda_thread_node &node, bool force = false) {
    store_counter_data(name, ctx, end, (double)(value), node, force);
}

static const char * getMemcpyKindString(uint8_t kind)
{
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "Memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "Memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "Memcpy HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "Memcpy AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "Memcpy AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "Memcpy AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "Memcpy DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "Memcpy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "Memcpy HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "Memcpy PtoP";
        default:
            break;
    }
    return "<unknown>";
}

static const char *
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
    switch (kind)
    {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            return "Unified Memcpy HTOD";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            return "Unified Memcpy DTOH";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            return "Unified Memory CPU Page Fault";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
            return "Unified Memory GPU Page Fault";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
            return "Unified Memory Trashing";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
            return "Unified Memory Throttling";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP:
            return "Unified Memory Remote Map";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
            return "Unified Memory DTOD";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT:
            return "Unified Memory Count";
        default:
            break;
    }
    return "<unknown>";
}

static uint32_t
getUvmCounterDevice(CUpti_ActivityUnifiedMemoryCounterKind kind,
        uint32_t source, uint32_t dest)
{
    switch (kind)
    {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            return dest;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            return source;
        default:
            break;
    }
    return 0;
}

//array enumerating CUpti_OpenAccEventKind strings
const char* openacc_event_names[] = {
    "OpenACC invalid event", // "CUPTI_OPENACC_EVENT_KIND_INVALD",
    "OpenACC device init", // "CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT",
    "OpenACC device shutdown", // "CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN",
    "OpenACC runtime shutdown", // "CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN",
    "OpenACC enqueue launch", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH",
    "OpenACC enqueue upload", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD",
    "OpenACC enqueue download", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD",
    "OpenACC wait", // "CUPTI_OPENACC_EVENT_KIND_WAIT",
    "OpenACC implicit wait", // "CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT",
    "OpenACC compute construct", // "CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT",
    "OpenACC update", // "CUPTI_OPENACC_EVENT_KIND_UPDATE",
    "OpenACC enter data", // "CUPTI_OPENACC_EVENT_KIND_ENTER_DATA",
    "OpenACC exit data", // "CUPTI_OPENACC_EVENT_KIND_EXIT_DATA",
    "OpenACC create", // "CUPTI_OPENACC_EVENT_KIND_CREATE",
    "OpenACC delete", // "CUPTI_OPENACC_EVENT_KIND_DELETE",
    "OpenACC alloc", // "CUPTI_OPENACC_EVENT_KIND_ALLOC",
    "OpenACC free" // "CUPTI_OPENACC_EVENT_KIND_FREE"
};

const char* openmp_event_names[] = {
    "OpenMP Invalid", // CUPTI_OPENMP_EVENT_KIND_INVALID
    "OpenMP Parallel", // CUPTI_OPENMP_EVENT_KIND_PARALLEL
    "OpenMP Task", // CUPTI_OPENMP_EVENT_KIND_TASK
    "OpenMP Thread", // CUPTI_OPENMP_EVENT_KIND_THREAD
    "OpenMP Idle", // CUPTI_OPENMP_EVENT_KIND_IDLE
    "OpenMP Wait Barrier", // CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER
    "OpenMP Wait Taskwait" // CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT
};

const char * sync_event_names[] = {
    "Unknown", // CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN = 0
    "Event Synchronize", // CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE = 1
    "Stream Wait", // CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT = 2
    "Stream Synchronize", // CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE = 3
    "Context Synchronize" // CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE = 4
};

#if 0
static const char * getComputeApiKindString(uint16_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
            return "CUDA";
        case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
            return "CUDA_MPS";
        default:
            break;
    }
    return "<unknown>";
}

static void deviceActivity(CUpti_Activity *record) {
    CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
    printf("DEVICE %s (%u), capability %u.%u,\n"
           "\tglobal memory (bandwidth %u GB/s, size %u MB),\n"
           "\tmultiprocessors %u, clock %u MHz\n",
           device->name, device->id,
           device->computeCapabilityMajor,
           device->computeCapabilityMinor,
           (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
           (unsigned int) (device->globalMemorySize / 1024 / 1024),
           device->numMultiprocessors,
           (unsigned int) (device->coreClockRate / 1000));
}

static void deviceAttributeActivity(CUpti_Activity *record) {
    CUpti_ActivityDeviceAttribute *attribute =
        (CUpti_ActivityDeviceAttribute *)record;
    printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
            attribute->attribute.cupti, attribute->deviceId,
            (unsigned long long)attribute->value.vUint64);
}

static void contextActivity(CUpti_Activity *record) {
    CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
    printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
            context->contextId, context->deviceId,
            getComputeApiKindString(context->computeApiKind),
            context->nullStreamId);
}
#endif

static void memcpyActivity2(CUpti_Activity *record) {
    CUpti_ActivityMemcpy2 *memcpy = (CUpti_ActivityMemcpy2 *) record;
    std::stringstream ss;
    ss << getMemcpyKindString(memcpy->copyKind) << " "
       << memcpy->deviceId << "->" << memcpy->dstDeviceId;
    std::string name{ss.str()};
    apex::cuda_thread_node node(memcpy->deviceId, memcpy->contextId,
        memcpy->streamId, APEX_ASYNC_MEMORY);
    store_profiler_data(name, memcpy->correlationId, memcpy->start,
            memcpy->end, node, "DataFlow",
            memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH);
    if (apex::apex_options::use_cuda_counters()) {
        store_counter_data("GPU: Bytes", name, memcpy->end,
            memcpy->bytes, node, true);
        // (1024 * 1024 * 1024) / 1,000,000,000
        // constexpr double GIGABYTES{1.073741824};
        double duration = (double)(memcpy->end - memcpy->start);
        double gbytes = (double)(memcpy->bytes); // / GIGABYTES;
        // dividing bytes by nanoseconds should give us GB/s
        double bandwidth = gbytes / duration;
        store_counter_data("GPU: Bandwidth (GB/s)", name,
            memcpy->end, bandwidth, node, true);
    }
}

static void memcpyActivity(CUpti_Activity *record) {
    CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
    if (memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD) {
        return memcpyActivity2(record);
    }
    std::string name{getMemcpyKindString(memcpy->copyKind)};
    apex::cuda_thread_node node(memcpy->deviceId, memcpy->contextId,
        memcpy->streamId, APEX_ASYNC_MEMORY);
    store_profiler_data(name, memcpy->correlationId, memcpy->start,
            memcpy->end, node, "DataFlow",
            memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH);
    if (apex::apex_options::use_cuda_counters()) {
        store_counter_data("GPU: Bytes", name, memcpy->end,
            memcpy->bytes, node, true);
        // (1024 * 1024 * 1024) / 1,000,000,000
        // constexpr double GIGABYTES{1.073741824};
        double duration = (double)(memcpy->end - memcpy->start);
        double gbytes = (double)(memcpy->bytes); // / GIGABYTES;
        // dividing bytes by nanoseconds should give us GB/s
        double bandwidth = gbytes / duration;
        store_counter_data("GPU: Bandwidth (GB/s)", name,
            memcpy->end, bandwidth, node, true);
    }
}

static void unifiedMemoryActivity(CUpti_Activity *record) {
    CUpti_ActivityUnifiedMemoryCounter2 *memcpy =
        (CUpti_ActivityUnifiedMemoryCounter2 *) record;
    std::string name{getUvmCounterKindString(memcpy->counterKind)};
    uint32_t device = getUvmCounterDevice(
            (CUpti_ActivityUnifiedMemoryCounterKind) memcpy->counterKind,
            memcpy->srcId, memcpy->dstId);
    apex::cuda_thread_node node(device, 0, 0, APEX_ASYNC_MEMORY);
    if (memcpy->counterKind ==
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
            || memcpy->counterKind ==
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH) {
        // The context isn't available, and the streamID isn't valid
        // (per CUPTI documentation)
        store_profiler_data(name, 0, memcpy->start, memcpy->end, node, "DataFlow");
        if (apex::apex_options::use_cuda_counters()) {
            store_counter_data("GPU: Bytes", name, memcpy->end,
                    memcpy->value, node, true);
            // (1024 * 1024 * 1024) / 1,000,000,000
            constexpr double GIGABYTES{1.073741824};
            double duration = (double)(memcpy->end - memcpy->start);
            double gbytes = (double)(memcpy->value) / GIGABYTES;
            // dividing bytes by nanoseconds should give us GB/s
            double bandwidth = gbytes / duration;
            store_counter_data("GPU: Bandwidth (GB/s)", name,
                    memcpy->end, bandwidth, node, true);
        }
    } else if (memcpy->counterKind ==
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING) {
        store_profiler_data(name, 0, memcpy->start, memcpy->end, node, "DataFlow");
    } else if (memcpy->counterKind ==
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT) {
        store_profiler_data(name, 0, memcpy->start, memcpy->end, node, "DataFlow");
        store_counter_data("Groups for same page", name, memcpy->end,
                memcpy->value, node);
    /*
    } else if (memcpy->counterKind ==
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT) {
        store_counter_data(nullptr, name, memcpy->start,
                1, node);
    */
    }
}

static void memsetActivity(CUpti_Activity *record) {
    CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
    const std::string name{"Memset"};
    apex::cuda_thread_node node(memset->deviceId, memset->contextId,
            memset->streamId, APEX_ASYNC_MEMORY);
    store_profiler_data(name, memset->correlationId, memset->start,
            memset->end, node, "DataFlow");
}

static void kernelActivity(CUpti_Activity *record) {
    CUpti_ActivityKernel4 *kernel =
        (CUpti_ActivityKernel4 *) record;
    std::string tmp = std::string(kernel->name);
    //DEBUG_PRINT("Kernel CorrelationId: %u\n", kernel->correlationId);
    apex::cuda_thread_node node(kernel->deviceId, kernel->contextId,
            kernel->streamId, APEX_ASYNC_KERNEL);
    store_profiler_data(tmp, kernel->correlationId, kernel->start,
            kernel->end, node, "ControlFlow");
    if (apex::apex_options::use_cuda_counters()) {
        std::string demangled = apex::demangle(kernel->name);
        store_counter_data("GPU: Dynamic Shared Memory (B)",
                demangled, kernel->end, kernel->dynamicSharedMemory, node);
        store_counter_data("GPU: Local Memory Per Thread (B)",
                demangled, kernel->end, kernel->localMemoryPerThread, node);
        store_counter_data("GPU: Local Memory Total (B)",
                demangled, kernel->end, kernel->localMemoryTotal, node);
        store_counter_data("GPU: Registers Per Thread",
                demangled, kernel->end, kernel->registersPerThread, node);
        store_counter_data("GPU: Shared Memory Size (B)",
                demangled, kernel->end, kernel->sharedMemoryExecuted, node);
        store_counter_data("GPU: Static Shared Memory (B)",
                demangled, kernel->end, kernel->staticSharedMemory, node);
        /* Get grid and block values */
        if (apex::apex_options::use_cuda_kernel_details()) {
            store_counter_data("GPU: blockX",
                demangled, kernel->end, kernel->blockX, node);
            store_counter_data("GPU: blockY",
                demangled, kernel->end, kernel->blockY, node);
            store_counter_data("GPU: blockZ",
                demangled, kernel->end, kernel->blockZ, node);
            store_counter_data("GPU: gridX",
                demangled, kernel->end, kernel->gridX, node);
            store_counter_data("GPU: gridY",
                demangled, kernel->end, kernel->gridY, node);
            store_counter_data("GPU: gridZ",
                demangled, kernel->end, kernel->gridZ, node);
            if (kernel->queued != CUPTI_TIMESTAMP_UNKNOWN) {
                store_counter_data("GPU: queue delay (us)",
                    demangled, kernel->end,
                    (kernel->start - kernel->queued)*1.0e-3, node);
            }
            if (kernel->submitted != CUPTI_TIMESTAMP_UNKNOWN) {
                store_counter_data("GPU: submit delay (us)",
                    demangled, kernel->end,
                    (kernel->start - kernel->submitted)*1.0e-3, node);
            }
        }
    }
}

static void openaccDataActivity(CUpti_Activity *record) {
    CUpti_ActivityOpenAccData *data = (CUpti_ActivityOpenAccData *) record;
    std::string label{openacc_event_names[data->eventKind]};
    apex::cuda_thread_node node(data->cuDeviceId, data->cuContextId,
        data->cuStreamId, APEX_ASYNC_MEMORY);
    store_profiler_data(label, data->externalId, data->start, data->end, node, "DataFlow");
    const std::string bytes{"Bytes Transferred"};
    store_counter_data(label.c_str(), bytes, data->end, data->bytes, node);
}

static void openaccKernelActivity(CUpti_Activity *record) {
    CUpti_ActivityOpenAccLaunch *data = (CUpti_ActivityOpenAccLaunch *) record;
    std::string label{openacc_event_names[data->eventKind]};
    apex::cuda_thread_node node(data->cuDeviceId, data->cuContextId,
        data->cuStreamId, APEX_ASYNC_KERNEL);
    store_profiler_data(label, data->externalId, data->start,
            data->end, node, "ControlFlow");
    const std::string gangs{"Num Gangs"};
    store_counter_data(label.c_str(), gangs, data->end, data->numGangs, node);
    const std::string workers{"Num Workers"};
    store_counter_data(label.c_str(), workers, data->end, data->numWorkers, node);
    const std::string lanes{"Num Vector Lanes"};
    store_counter_data(label.c_str(), lanes, data->end, data->vectorLength, node);
}

static void openaccOtherActivity(CUpti_Activity *record) {
    CUpti_ActivityOpenAccOther *data = (CUpti_ActivityOpenAccOther *) record;
    std::string label{openacc_event_names[data->eventKind]};
    apex::cuda_thread_node node(data->cuDeviceId, data->cuContextId,
        data->cuStreamId, APEX_ASYNC_OTHER);
    store_profiler_data(label, data->externalId, data->start, data->end, node, "OtherFlow");
}

static void openmpActivity(CUpti_Activity *record) {
    CUpti_ActivityOpenMp *data = (CUpti_ActivityOpenMp *) record;
    std::string label{openmp_event_names[data->eventKind]};
}

static void syncActivity(CUpti_Activity *record) {
    // when tracking memory allocations, ignore the ones in cuda device synchronize
    CUpti_ActivitySynchronization *data =
        (CUpti_ActivitySynchronization *) record;
    /* Check whether there is timing information */
    if (data->start == 0 && data->end == 0) { return; }
    std::string label{sync_event_names[data->type]};
    uint32_t device = 0;
    map_mutex.lock();
    device = context_map[data->contextId];
    map_mutex.unlock();
    uint32_t context = data->contextId;
    uint32_t stream = 0;
    /* only these events have a stream ID */
    if (data->type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT ||
        data->type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE) {
        stream = data->streamId;
    }
    apex::cuda_thread_node node(device, context, stream, APEX_ASYNC_SYNCHRONIZE);
    /* Event Synchronize doesn't have a stream ID, and can come from any thread,
     * and can overlap.  So if we are OTF2 tracing, ignore them. */
    if (apex::apex_options::use_otf2() &&
        data->type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE) {
        store_profiler_data(label, data->correlationId, data->start, data->end, node, "SyncFlow", false, false);
    } else {
        store_profiler_data(label, data->correlationId, data->start, data->end, node, "SyncFlow");
    }
}

static void printActivity(CUpti_Activity *record) {
    //auto p = apex::scoped_timer("APEX: CUPTI printActivity");
    switch (record->kind)
    {
#if 0
        case CUPTI_ACTIVITY_KIND_DEVICE: {
            deviceActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
            deviceAttributeActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_CONTEXT: {
            contextActivity(record);
            break;
        }
#endif
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            memcpyActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
        {
            memcpyActivity2(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
            unifiedMemoryActivity(record);
            break;
        }
#if 0 // not until CUDA 11
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
        {
            CUpti_ActivityMemcpyPtoP *memcpy = (CUpti_ActivityMemcpyPtoP *) record;
            break;
        }
#endif
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            memsetActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
        {
            kernelActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OPENACC_DATA: {
            DEBUG_PRINT("OpenACC Data!\n");
            openaccDataActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH: {
            DEBUG_PRINT("OpenACC Launch!\n");
            openaccKernelActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OPENACC_OTHER: {
            DEBUG_PRINT("OpenACC Other!\n");
            openaccOtherActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OPENMP: {
            openmpActivity(record);
            break;
        }
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            syncActivity(record);
            break;
        }
        default:
#if 0
            printf("  <unknown>\n");
#endif
            break;
    }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
    size_t *maxNumRecords)
{
    // when tracking memory allocations, ignore these
    apex::in_apex prevent_nonsense;
    num_buffers++;
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
    uint8_t *buffer, size_t size, size_t validSize)
{
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { free(buffer); return; }
    //auto p = apex::scoped_timer("APEX: CUPTI Buffer Completed");
    //printf("%s...", __APEX_FUNCTION__); fflush(stdout);
    static bool registered = register_myself(false);
    // when tracking memory allocations, ignore these
    apex::in_apex prevent_nonsense;
    num_buffers_processed++;
    if (flushing) { std::cout << "." << std::flush; }
    APEX_UNUSED(registered);
    CUptiResult status;
    CUpti_Activity *record = NULL;
    APEX_UNUSED(size);

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                printActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
    }

    free(buffer);
    //printf("done."); fflush(stdout);
}

/* this has to happen AFTER cuInit(). */
void configureUnifiedMemorySupport(void) {
    CUpti_ActivityUnifiedMemoryCounterConfig config[4];
    int num_counters = 2;

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    if (apex::apex_options::use_cuda_counters()) {
        num_counters = 4;
        config[2].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
        config[2].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
        config[2].deviceId = 0;
        config[2].enable = 1;

        config[3].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
        config[3].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT;
        config[3].deviceId = 0;
        config[3].enable = 1;
    }

    CUptiResult res = cuptiActivityConfigureUnifiedMemoryCounter(config, num_counters);
    if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) {
        printf("Test is waived, unified memory is not supported on the underlying platform.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
        printf("Test is waived, unified memory is not supported on the device.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
        printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
    }
    else {
        CUPTI_CALL(res);
    }
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)); // 25
}

bool initialize_first_time() {
    apex::init("APEX CUDA support", 0, 1);
    configureUnifiedMemorySupport();
    return true;
}

// void(CUDART_CB* cudaHostFn_t )( void*  userData )
void kernelComplete(void* userData) {
    char * name = (char *)(userData);
    std::cout << name << " complete." << std::endl;
    free(name);
}

void notifyKernelComplete(CUpti_CallbackId id, const void* params, const char * symbolName) {
    cudaStream_t stream;
    switch (id) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
        {
            stream = ((cudaLaunchKernel_v7000_params_st*)(params))->stream;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
        {
            stream = ((cudaLaunchKernel_ptsz_v7000_params_st*)(params))->stream;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeGetParams_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddKernelNode_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010:
        default:
            return;
    }
    CUDA_CALL(cudaLaunchHostFunc(stream, kernelComplete, (void*)(strdup(symbolName))));
    return;
}

bool isLaunch(CUpti_CallbackId id) {
    if (id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000 ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
        id == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000 ||
        id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
        id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz
        ) {
        return true;
    }
    return false;
}

bool getBytesIfMalloc(CUpti_CallbackId id, const void* params, std::string context, bool isEnter) {
    size_t bytes = 0;
    bool onHost = false;
    bool managed = false;
    void* ptr = nullptr;
    static std::atomic<size_t> totalAllocated{0};
    static std::unordered_map<void*,size_t> memoryMap;
    std::mutex mapMutex;
    static std::atomic<size_t> hostTotalAllocated{0};
    static std::unordered_map<void*,size_t> hostMemoryMap;
    std::mutex hostMapMutex;
    bool free = false;
    if (apex::apex_options::use_cuda_driver_api() ||
        apex::apex_options::use_cuda_kernel_details()) {
        switch (id) {
            case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc: {
                size_t tmp = (size_t)*((cuMemAlloc_params_st*)(params))->dptr;
                ptr = (void*)(tmp);
                bytes = ((cuMemAlloc_params_st*)(params))->bytesize;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch: {
                ptr = ((cuMemAllocPitch_params_st*)(params))->dptr;
                bytes = ((cuMemAllocPitch_params_st*)(params))->WidthInBytes *
                        ((cuMemAllocPitch_params_st*)(params))->Height *
                        ((cuMemAllocPitch_params_st*)(params))->ElementSizeBytes;
                break;
            }
            /*
            case CUPTI_DRIVER_TRACE_CBID_cu64MemAlloc: {
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cu64MemAllocPitch: {
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cu64MemHostAlloc: {
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2: {
                bytes = ((cuMemHostAlloc_v2_params_st*)(params))->bytesize;
                break;
            }
            */
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost: {
                ptr = *((cuMemAllocHost_params_st*)(params))->pp;
                bytes = ((cuMemAllocHost_params_st*)(params))->bytesize;
                onHost = true;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc: {
                ptr = *((cuMemHostAlloc_params_st*)(params))->pp;
                bytes = ((cuMemHostAlloc_params_st*)(params))->bytesize;
                onHost = true;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2: {
                ptr = (void*)*((cuMemAlloc_v2_params_st*)(params))->dptr;
                bytes = ((cuMemAlloc_v2_params_st*)(params))->bytesize;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2: {
                ptr = (void*)*((cuMemAllocPitch_v2_params_st*)(params))->dptr;
                bytes = ((cuMemAllocPitch_v2_params_st*)(params))->WidthInBytes *
                        ((cuMemAllocPitch_v2_params_st*)(params))->Height *
                        ((cuMemAllocPitch_v2_params_st*)(params))->ElementSizeBytes;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2: {
                ptr = *((cuMemAllocHost_v2_params_st*)(params))->pp;
                bytes = ((cuMemAllocHost_v2_params_st*)(params))->bytesize;
                onHost = true;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged: {
                ptr = (void*)*((cuMemAllocManaged_params_st*)(params))->dptr;
                bytes = ((cuMemAllocManaged_params_st*)(params))->bytesize;
                managed = true;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2: {
                ptr = (void*)((cuMemFree_v2_params_st*)(params))->dptr;
                free = true;
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost: {
                ptr = ((cuMemFreeHost_params_st*)(params))->p;
                free = true;
                onHost = true;
                break;
            }
#ifdef CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree
            case CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree: {
                ptr = (void*)((cuMemAddressFree_params_st*)(params))->ptr;
                free = true;
                break;
            }
#endif
            case CUPTI_DRIVER_TRACE_CBID_cuMemFree: {
                size_t tmp = (size_t)((cuMemFree_params_st*)(params))->dptr;
                ptr = (void*)(tmp);
                free = true;
                break;
            }
            default: {
                // return false;
                break;
            }
        }
    }
    if (!apex::apex_options::use_cuda_driver_api() ||
        apex::apex_options::use_cuda_kernel_details()) {
        switch (id) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
                ptr = *((cudaMalloc_v3020_params_st*)(params))->devPtr;
                bytes = ((cudaMalloc_v3020_params_st*)(params))->size;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020: {
                ptr = *((cudaMallocPitch_v3020_params_st*)(params))->devPtr;
                bytes = ((cudaMallocPitch_v3020_params_st*)(params))->width *
                        ((cudaMallocPitch_v3020_params_st*)(params))->height;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020: {
                ptr = ((cudaMallocArray_v3020_params_st*)(params))->array;
                bytes = ((cudaMallocArray_v3020_params_st*)(params))->width *
                        ((cudaMallocArray_v3020_params_st*)(params))->height;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020: {
                bytes = ((cudaHostAlloc_v3020_params_st*)(params))->size;
                ptr = *((cudaHostAlloc_v3020_params_st*)(params))->pHost;
                onHost = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020: {
                bytes = ((cudaMallocHost_v3020_params_st*)(params))->size;
                onHost = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020: {
                ptr = ((cudaMalloc3D_v3020_params_st*)(params))->pitchedDevPtr->ptr;
                cudaExtent extent = ((cudaMalloc3D_v3020_params_st*)(params))->extent;
                bytes = extent.depth * extent.height * extent.width;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020: {
                ptr = ((cudaMalloc3DArray_v3020_params_st*)(params))->array;
                cudaExtent extent = ((cudaMalloc3DArray_v3020_params_st*)(params))->extent;
                bytes = extent.depth * extent.height * extent.width;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000: {
                ptr = ((cudaMallocMipmappedArray_v5000_params_st*)(params))->mipmappedArray;
                cudaExtent extent = ((cudaMallocMipmappedArray_v5000_params_st*)(params))->extent;
                bytes = extent.depth * extent.height * extent.width;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000: {
                ptr = *((cudaMallocManaged_v6000_params_st*)(params))->devPtr;
                bytes = ((cudaMallocManaged_v6000_params_st*)(params))->size;
                managed = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020: {
                ptr = ((cudaFree_v3020_params*)(params))->devPtr;
                free = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020: {
                ptr = ((cudaFreeArray_v3020_params*)(params))->array;
                free = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020: {
                ptr = ((cudaFreeHost_v3020_params*)(params))->ptr;
                free = true;
                onHost = true;
                break;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000: {
                ptr = ((cudaFreeMipmappedArray_v5000_params*)(params))->mipmappedArray;
                free = true;
                break;
            }
            default: {
                return false;
            }
        }
    }
    // If we are in the enter of a function, and we are freeing memory,
    // then update and record the bytes allocated
    if (free && isEnter) {
        double value = 0;
        //std::cout << "Freeing " << ptr << std::endl;
        if (onHost) {
            hostMapMutex.lock();
            if (hostMemoryMap.count(ptr) > 0) {
                bytes = hostMemoryMap[ptr];
                hostMemoryMap.erase(ptr);
            } else {
                hostMapMutex.unlock();
                return false;
            }
            hostMapMutex.unlock();
            value = (double)(bytes);
            store_sync_counter_data("Host: Page-locked Bytes Freed", context, value);
            hostTotalAllocated.fetch_sub(bytes, std::memory_order_relaxed);
            value = (double)(hostTotalAllocated);
            store_sync_counter_data("GPU: Total Bytes Occupied on Host", context, value, false, false);
            apex::recordFree(ptr);
        } else {
            mapMutex.lock();
            if (memoryMap.count(ptr) > 0) {
                bytes = memoryMap[ptr];
                memoryMap.erase(ptr);
            } else {
                mapMutex.unlock();
                return false;
            }
            mapMutex.unlock();
            value = (double)(bytes);
            if (managed) {
                store_sync_counter_data("GPU: Bytes Freed (Managed)", context, value);
            } else {
                store_sync_counter_data("GPU: Bytes Freed", context, value);
            }
            totalAllocated.fetch_sub(value, std::memory_order_relaxed);
            value = (double)(totalAllocated);
            store_sync_counter_data("GPU: Total Bytes Occupied on Device", context, value, false, false);
            apex::recordFree(ptr, false);
        }
    // If we are in the exit of a function, and we are allocating memory,
    // then update and record the bytes allocated
    } else if (!free && !isEnter) {
        if (bytes == 0) return false;
        double value = (double)(bytes);
        //std::cout << "Allocating " << value << " bytes at " << ptr << std::endl;
        if (onHost) {
            store_sync_counter_data("Host: Page-locked Bytes Allocated", context, value);
            hostMapMutex.lock();
            hostMemoryMap[ptr] = value;
            hostMapMutex.unlock();
            hostTotalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            value = (double)(hostTotalAllocated);
            store_sync_counter_data("GPU: Total Bytes Occupied on Host", context, value, false, false);
            apex::recordAlloc(bytes, ptr, APEX_GPU_DEVICE_MALLOC);
            return true;
        } else {
            if (managed) {
                store_sync_counter_data("GPU: Bytes Allocated (Managed)", context, value);
            } else {
                store_sync_counter_data("GPU: Bytes Allocated", context, value);
            }
            mapMutex.lock();
            memoryMap[ptr] = value;
            mapMutex.unlock();
            totalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            value = (double)(totalAllocated);
            store_sync_counter_data("GPU: Total Bytes Occupied on Device", context, value, false, false);
            apex::recordAlloc(bytes, ptr, APEX_GPU_DEVICE_MALLOC, false);
        }
    }
    return true;
}

void register_new_context(const void *params) {
    //printf("New Context\n");
    APEX_UNUSED(params);
    if (apex::apex_options::use_cuda_kernel_activity()) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)); // 10
    }
    if (apex::apex_options::use_cuda_memory_activity()) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)); // 1
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2)); // 22
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET)); // 2
    }
    if (apex::apex_options::use_cuda_sync_activity()) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)); // 38
    }
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA)); // 33
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH)); // 34
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER)); // 35
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENMP)); // 47
#if 0
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)); // 3   <- disables concurrency
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER)); // 4
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)); // 5
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EVENT)); // 6
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_METRIC)); // 7
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE)); // 8
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT)); // 9
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME)); // 11
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER)); // 12
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA)); // 13
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR)); // 14
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS)); // 15
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_BRANCH)); // 16
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD)); // 17
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CDP_KERNEL)); // 18
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PREEMPTION)); // 19
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT)); // 20
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EVENT_INSTANCE)); // 21
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_METRIC_INSTANCE)); // 23
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION)); // 24
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)); // 25
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION)); // 26
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MODULE)); // 27
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE)); // 28
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SHARED_ACCESS)); // 29
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING)); // 30
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO)); // 31
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION)); // 32
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CUDA_EVENT)); // 36
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_STREAM)); // 37
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)); // 38
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)); // 39
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NVLINK)); // 40
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT)); // 41
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE)); // 42
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC)); // 43
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE)); // 44
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY)); // 45
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PCIE)); // 46
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API)); // 48
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_COUNT)); // 49
#endif
}

std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib) {
    std::string tmp;
    if (eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII) {
        tmp = std::string(eventAttrib->message.ascii);
    } else {
        std::wstring wtmp(eventAttrib->message.unicode);
        tmp = std::string(wtmp.begin(), wtmp.end());
    }
    return tmp;
}

double get_nvtx_payload(const nvtxEventAttributes_t * eventAttrib) {
    double payload;
    switch (eventAttrib->payloadType) {
        case NVTX_PAYLOAD_TYPE_UNSIGNED_INT64:
        {
            payload = (double)eventAttrib->payload.ullValue;
            break;
        }
        case NVTX_PAYLOAD_TYPE_INT64:
        {
            payload = (double)eventAttrib->payload.llValue;
            break;
        }
        case NVTX_PAYLOAD_TYPE_DOUBLE:
        {
            payload = (double)eventAttrib->payload.dValue;
            break;
        }
        case NVTX_PAYLOAD_TYPE_UNSIGNED_INT32:
        {
            payload = (double)eventAttrib->payload.uiValue;
            break;
        }
        case NVTX_PAYLOAD_TYPE_INT32:
        {
            payload = (double)eventAttrib->payload.iValue;
            break;
        }
        case NVTX_PAYLOAD_TYPE_FLOAT:
        {
            payload = (double)eventAttrib->payload.fValue;
            break;
        }
        default:
        {
            payload = 0.0;
            break;
        }
    }
    return payload;
}

void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata) {
    // disable memory management tracking in APEX during this callback
    apex::in_apex prevent_deadlocks;


    /* Unfortunately, when ranges are started/ended, they can overlap.
     * Unlike push/pop, which are a true stack.  Even worse, CUDA/CUPTI
     * doesn't give us any way to tie the start with the end - the start
     * is identified with a string, and the end is identified with an
     * index.  We can't match them up. So, we'll just treat them as a
     * stack for safety.  Also - ranges can start/end on different threads!
     * So that means we have to use a common stack and hope for the best?
     * Maybe we should just ignore the range start/end events.
     */

    const CUpti_NvtxData *nvtxInfo = (CUpti_NvtxData *)cbdata;

    switch (id) {
        /* Domain events */
        case CUPTI_CBID_NVTX_nvtxDomainCreateA:
        case CUPTI_CBID_NVTX_nvtxDomainCreateW:
        {
            /* nothing to do - handled in the wrapper. */
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainDestroy:
        {
            nvtxDomainDestroy_params *params =
                (nvtxDomainDestroy_params *)nvtxInfo->functionParams;
            get_domain_map().erase(params->domain);
            break;
        }
        /* Range start events */
        case CUPTI_CBID_NVTX_nvtxRangeStartEx:
        case CUPTI_CBID_NVTX_nvtxRangeStartA:
        case CUPTI_CBID_NVTX_nvtxRangeStartW:
        {
            /* nothing to do, handled in the wrapper */
            break;
        }
#ifndef APEX_BROKEN_CUPTI_NVTX_PUSH_POP
        case CUPTI_CBID_NVTX_nvtxDomainRangeEnd:
        {
            /* Range start/end is too risky for OTF2 */
            if (apex::apex_options::use_otf2()) { break; }
            nvtxDomainRangeEnd_params *params =
                (nvtxDomainRangeEnd_params *)nvtxInfo->functionParams;
            apex::stop(get_range_map()[params->core.id]);
            get_range_map().erase(params->core.id);
            break;
        }
        /* Range end events */
        case CUPTI_CBID_NVTX_nvtxRangeEnd:
        {
            /* Range start/end is too risky for OTF2 */
            if (apex::apex_options::use_otf2()) { break; }
            nvtxRangeEnd_params *params =
                (nvtxRangeEnd_params *)nvtxInfo->functionParams;
            apex::stop(get_range_map()[params->id]);
            get_range_map().erase(params->id);
            break;
        }
        /* Range push events */
        case CUPTI_CBID_NVTX_nvtxRangePushA: {
            nvtxRangePushA_params *params =
                (nvtxRangePushA_params *)nvtxInfo->functionParams;
            std::string tmp(params->message);
            auto timer = apex::new_task(tmp);
            apex::start(timer);
            get_range_stack().push(timer);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushW: {
            nvtxRangePushW_params *params =
                (nvtxRangePushW_params *)nvtxInfo->functionParams;
            std::wstring wtmp(params->message);
            std::string tmp(wtmp.begin(), wtmp.end());
            auto timer = apex::new_task(tmp);
            apex::start(timer);
            get_range_stack().push(timer);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushEx: {
            nvtxRangePushEx_params *params =
                (nvtxRangePushEx_params *)nvtxInfo->functionParams;
            std::string tmp = get_nvtx_message(params->eventAttrib);
            auto timer = apex::new_task(tmp);
            apex::start(timer);
            get_range_stack().push(timer);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
            nvtxDomainRangePushEx_params *params =
                (nvtxDomainRangePushEx_params *)nvtxInfo->functionParams;
            std::string tmp;
            if (params->domain != NULL) {
                std::string domain(get_domain_map()[params->domain]);
                std::stringstream ss;
                ss << domain << ": " << get_nvtx_message(params->core.eventAttrib);
                tmp = ss.str();
            } else {
                tmp = get_nvtx_message(params->core.eventAttrib);
            }
            auto timer = apex::new_task(tmp);
            apex::start(timer);
            get_range_stack().push(timer);
            break;
        }
        /* Range pop events */
        case CUPTI_CBID_NVTX_nvtxRangePop:
        {
            if (!get_range_stack().empty()) {
                auto timer = get_range_stack().top();
                apex::stop(timer);
                get_range_stack().pop();
            }
            break;
        }
#endif
        case CUPTI_CBID_NVTX_nvtxMarkA:
        {
            /* marker event with dummy value */
            nvtxMarkA_params *params =
                (nvtxMarkA_params *)nvtxInfo->functionParams;
            std::string tmp(params->message);
            store_sync_counter_data(nullptr, tmp, 0);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxMarkW:
        {
            /* marker event with dummy value */
            nvtxMarkW_params *params =
                (nvtxMarkW_params *)nvtxInfo->functionParams;
            std::wstring wtmp(params->message);
            std::string tmp(wtmp.begin(), wtmp.end());
            store_sync_counter_data(nullptr, tmp, 0);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxMarkEx:
        {
            nvtxMarkEx_params *params =
                (nvtxMarkEx_params *)nvtxInfo->functionParams;
            std::string tmp = get_nvtx_message(params->eventAttrib);
            double payload = get_nvtx_payload(params->eventAttrib);
            store_sync_counter_data(nullptr, tmp, payload);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx:
        {
            nvtxDomainMarkEx_params *params =
                (nvtxDomainMarkEx_params *)nvtxInfo->functionParams;
            std::string tmp = get_nvtx_message(params->core.eventAttrib);
            double payload = get_nvtx_payload(params->core.eventAttrib);
            if (params->domain != NULL) {
                std::string domain(get_domain_map()[params->domain]);
                store_sync_counter_data(domain.c_str(), tmp, payload, true);
            } else {
                store_sync_counter_data(nullptr, tmp, payload);
            }
            break;
        }
        default: {
            break;
        }
    }
}

bool ignoreMalloc(CUpti_CallbackDomain domain,
    CUpti_ApiCallbackSite site, CUpti_CallbackId id) {
    static bool ignore{true};
    // this is a one way switch, when changed it never changes again
    // We want to ignore memory tracking on only the first CUPTI Runtime
    // call, because CUPTI leaks so much memory!
    if (ignore) {
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            if (site == CUPTI_API_ENTER) {
                //printf("Ignoring on event %d\n", id);
                apex::in_apex::get()++;
            } else if (site == CUPTI_API_EXIT) {
                //printf("Ignoring on event %d\n", id);
                apex::in_apex::get()--;
                // All of the events under the code of cudaMalloc don't
                // do much, so until we see a "real" API call that does
                // significant work, ignore memory accesses.
                if (id > CUPTI_RUNTIME_TRACE_CBID_cudaSetDeviceFlags_v3020) {
                    ignore = false;
                }
            }
        }
    }
    return ignore;
}

void apex_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain,
        CUpti_CallbackId id, const void *params) {
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }
    // disable memory management tracking in APEX during this callback
    apex::in_apex prevent_deadlocks;
    static bool initialized = initialize_first_time();
    APEX_UNUSED(initialized);
    static APEX_NATIVE_TLS bool registered = register_myself(true);
    APEX_UNUSED(registered);
    /* Supposedly, we can use the ud or cbdata->contextData fields
     * to pass data from the start to the end event, but it isn't
     * broadly supported in the CUPTI interface, so we'll manage the
     * timer stack locally. */
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    APEX_UNUSED(ud);
    APEX_UNUSED(id);
    APEX_UNUSED(domain);
    //printf("Callback: %d, %d\n", domain, id);
    if (!apex::thread_instance::is_worker()) { return; }
    if (params == NULL) { return; }

    /* Check for exit */
    if (!allGood) {
	return;
    }

    /* Check for a new context */
    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
        if (id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            register_new_context(params);
        }
        //return;
    }

    /* Check for user-level instrumentation */
    if (domain == CUPTI_CB_DOMAIN_NVTX) {
        handle_nvtx_callback(id, params);
        return;
    }

    CUpti_CallbackData * cbdata = (CUpti_CallbackData*)(params);

    // sadly, CUPTI leaks a lot of memory from the first runtime API call.
    if (apex::apex_options::track_gpu_memory()) {
        ignoreMalloc(domain, cbdata->callbackSite, id);
    }

    if (cbdata->callbackSite == CUPTI_API_ENTER) {
        std::stringstream ss;
        ss << cbdata->functionName;
        if (apex::apex_options::use_cuda_kernel_details() && isLaunch(id)) {
            // protect against stack overflows from cupti. yes, it happens.
            if (cbdata->symbolName != NULL && cbdata->symbolName > params &&
                strlen(cbdata->symbolName) > 0) {
                ss << ": " << cbdata->symbolName;
            }
        }
        std::string tmp(ss.str());
        /*
           std::string tmp(cbdata->functionName);
           */
        auto timer = apex::new_task(tmp);
	if (timer == nullptr) {
            /* This happens when we've hit finalize but there are still some
	     * CUDA calls that come in. Ignore it.
	     */
            return;
	}
        apex::start(timer);
        timer_stack.push(timer);
        apex::async_event_data as_data(timer->prof->get_start_us(),
            "OtherFlow", cbdata->correlationId,
            apex::thread_instance::get_id(), cbdata->functionName);
        map_mutex.lock();
        correlation_map[cbdata->correlationId] = timer;
        correlation_kernel_data_map[cbdata->correlationId] = as_data;
        map_mutex.unlock();
        getBytesIfMalloc(id, cbdata->functionParams, tmp, true);
    } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
        /* Not sure how to use this yet... if this is a kernel launch, we can
         * run a function on the host, launched from the stream.  That gives us
         * a synchronous callback to tell us an event when the kernel finished.
         */
        /*
        if (cbdata->callbackSite == CUPTI_API_EXIT &&
            cbdata->functionName != nullptr &&
            cbdata->symbolName != nullptr)
            notifyKernelComplete(id, cbdata->functionParams, cbdata->symbolName);
        */

        /* If this is a malloc/free, keep track of total bytes */
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            std::stringstream ss;
            ss << cbdata->functionName;
            if (apex::apex_options::use_cuda_kernel_details() && isLaunch(id)) {
                if (cbdata->symbolName != NULL && cbdata->symbolName > params &&
                    strlen(cbdata->symbolName) > 0) {
                    ss << ": " << cbdata->symbolName;
                }
            }
            std::string tmp(ss.str());
            getBytesIfMalloc(id, cbdata->functionParams, tmp, false);
        }

        if (!timer_stack.empty()) {
        auto timer = timer_stack.top();
        apex::stop(timer);

        map_mutex.lock();
        apex::async_event_data as_data =
            correlation_kernel_data_map[cbdata->correlationId];
        as_data.parent_ts_stop = timer->prof->get_stop_us();
        correlation_kernel_data_map[cbdata->correlationId] = as_data;
        map_mutex.unlock();

        timer_stack.pop();

        /* Check for SetDevice call! */
        if (id == CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020 ||
            id == CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent) {
            /* Can't trust the parameter!  It lies. */
            //int device = ((cudaSetDevice_v3020_params_st*)(params))->device;
            uint32_t device{0};
            uint32_t context{0};
            cuptiGetDeviceId(cbdata->context, &device);
            cuptiGetContextId(cbdata->context, &context);
            apex::nvml::monitor::activateDeviceIndex(device);
            map_mutex.lock();
            context_map[context] = device;
            map_mutex.unlock();
        }
        }
    }
}

extern "C" {

void apex_init_cuda_tracing() {
    if (!apex::apex_options::use_cuda()) { return; }
    // disable memory management tracking in APEX during this initialization
    apex::in_apex prevent_deadlocks;
    // make sure APEX doesn't re-register this thread
    bool& registered = get_registered();
    registered = true;

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity API.
    // Some attributes require to be set before any CUDA context is created to be effective,
    // e.g. to be applied to all device buffer allocations (see documentation).
    size_t attrValue = 0, attrValueSize = sizeof(size_t);
    CUPTI_CALL(cuptiActivityGetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
    //attrValue = attrValue / 4;
    attrValue = BUF_SIZE;
    CUPTI_CALL(cuptiActivitySetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    /*
    CUPTI_CALL(cuptiActivityGetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
    attrValue = attrValue * 4;
    CUPTI_CALL(cuptiActivitySetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
    */

    /* now that the activity is configured, subscribe to callback support, too. */
    CUPTI_CALL(cuptiSubscribe(&subscriber,
                (CUpti_CallbackFunc)apex_cupti_callback_dispatch, NULL));
    // get device callbacks
    if (apex::apex_options::use_cuda_runtime_api()) {
        CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    }
    if (apex::apex_options::use_cuda_driver_api()) {
        CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    }
    if (apex::apex_options::use_cuda_kernel_details()) {
        uint8_t enable = 1;
        CUPTI_CALL(cuptiActivityEnableLatencyTimestamps(enable));
    }

    // get user-added instrumentation
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX));
    // Make sure we see CUPTI_CBID_RESOURCE_CONTEXT_CREATED events!
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE));

    /* These events aren't begin/end callbacks, so no need to support them. */
    //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE));

    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    startTimestampCPU = apex::profiler::now_ns();
    cuptiGetTimestamp(&startTimestampGPU);
    startTimestampCPU += apex::profiler::now_ns();
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    deltaTimestamp = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    //printf("CPU: %ld\n", startTimestampCPU);
    //printf("GPU: %ld\n", startTimestampGPU);
    //printf("Delta computed to be: %ld\n", deltaTimestamp);
    allGood = true;
}

/* This is the global "shutdown" method for flushing the buffer.  This is
 * called from apex::finalize().  It's the only function in the CUDA support
 * that APEX will call directly. */
    void apex_flush_cuda_tracing(void) {
        if (!apex::apex_options::use_cuda()) { return; }
        if ((num_buffers_processed + 10) < num_buffers) {
            if (apex::apex::instance()->get_node_id() == 0) {
                flushing = true;
                std::cout << "Flushing remaining " << std::fixed
                    << num_buffers-num_buffers_processed << " of " << num_buffers
                    << " CUDA/CUPTI buffers..." << std::endl;
            }
        }
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
        if (flushing) {
            std::cout << std::endl;
        }
    }

    void apex_stop_cuda_tracing(void) {
        if (!apex::apex_options::use_cuda()) { return; }
        apex_flush_cuda_tracing();
        CUPTI_CALL(cuptiUnsubscribe(subscriber));
        uint32_t version{0};
        CUPTI_CALL(cuptiGetVersion(&version));
        // cupti 12 introduced a bug that was fixed in version 12.4.
        // see https://forums.developer.nvidia.com/t/cuda-profiler-tools-interface-cupti-for-cuda-toolkit-12-4-is-now-available/279799
        //printf("Cupti version: %d\n", version);
        if (version < 18 || version > 21) {
            CUPTI_CALL(cuptiFinalize());
        }
        // get_range_map().clear();
	//std::cout << "* * * * * * * * EXITING CUPTI SUPPORT * * * * * * * * * *" << std::endl;
    	allGood = false;
    }
} // extern "C"
