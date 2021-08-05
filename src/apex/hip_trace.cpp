/*
   Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
using namespace std;

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#include "apex_api.hpp"
#include "apex.hpp"
#include "async_thread_node.hpp"
#include "trace_event_listener.hpp"
#ifdef APEX_HAVE_OTF2
#include "otf2_listener.hpp"
#endif
#include <stack>
#include <mutex>
#include <map>

#include "global_constructor_destructor.h"
DEFINE_CONSTRUCTOR(init_tracing);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#include <roctracer_hsa.h>
#include <roctracer_kfd.h>
#include <roctracer_roctx.h>

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                    \
    do {                                                        \
        int err = call;                                         \
        if (err != 0) {                                         \
            fprintf(stderr, "%s\n", roctracer_error_string());  \
            abort();                                            \
        }                                                       \
    } while (0)

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestampGPU{0};
static uint64_t startTimestampCPU{0};
static int64_t deltaTimestamp{0};

bool run_once() {
    static bool once{false};
    if (once) { return once; }
    if (apex::apex_options::use_hip_kfd_api()) {
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
    }
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    startTimestampCPU = apex::profiler::now_ns();
    roctracer_get_timestamp(&startTimestampGPU);
    startTimestampCPU += apex::profiler::now_ns();
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    deltaTimestamp = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    //printf("HIP timestamp:      %lu\n", startTimestampGPU);
    //printf("CPU timestamp:      %lu\n", startTimestampCPU);
    //printf("HIP delta timestamp: %ld\n", deltaTimestamp);
    apex::init("APEX HIP wrapper", 0, 1);
    once = true;
    return once;
}

/* This is like the CUDA NVTX API.  User-added instrumentation for
 * ranges that can be pushed/popped on a stack (and common to a thread
 * of execution) or started/stopped (and can be started by one thread
 * and stopped by another).
 */
void handle_roctx(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    APEX_UNUSED(domain);
    APEX_UNUSED(arg);
    static bool once = run_once();
    APEX_UNUSED(once);
    static thread_local std::stack<apex::profiler*> timer_stack;
    static std::map<roctx_range_id_t, apex::profiler*> timer_map;
    static std::mutex map_lock;
    const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
    switch (cid) {
        case ROCTX_API_ID_roctxRangePushA:
            {
                std::stringstream ss;
                ss << "roctx: " << data->args.message;
                timer_stack.push(apex::start(ss.str()));
                break;
            }
        case ROCTX_API_ID_roctxRangePop:
            {
                apex::stop(timer_stack.top());
                timer_stack.pop();
                break;
            }
        case ROCTX_API_ID_roctxRangeStartA:
            {
                std::stringstream ss;
                ss << "roctx: " << data->args.message;
                apex::profiler* p = apex::start(ss.str());
                const std::lock_guard<std::mutex> guard(map_lock);
                timer_map.insert(
                        std::pair<roctx_range_id_t, apex::profiler*>(
                            data->args.id, p));
                break;
            }
        case ROCTX_API_ID_roctxRangeStop:
            {
                const std::lock_guard<std::mutex> guard(map_lock);
                auto p = timer_map.find(data->args.id);
                if (p != timer_map.end()) {
                    apex::stop(p->second);
                    timer_map.erase(data->args.id);
                }
                break;
            }
        case ROCTX_API_ID_roctxMarkA:
            // we do nothing with marker events...for now
        default:
            break;
    }
    return;
}

/* The map that holds correlation IDs and matches them to GUIDs */
std::unordered_map<uint32_t, std::shared_ptr<apex::task_wrapper>> correlation_map;
std::unordered_map<uint32_t, const void*> correlation_kernel_map;
std::mutex correlation_map_mutex;

/* This is the "low level" API - lots of events if interested. */
void handle_roc_kfd(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    APEX_UNUSED(domain);
    APEX_UNUSED(arg);
    static bool once = run_once();
    APEX_UNUSED(once);
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    const kfd_api_data_t* data = (const kfd_api_data_t*)(callback_data);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        auto timer = apex::new_task(
			roctracer_op_string(ACTIVITY_DOMAIN_KFD_API, cid, 0));
        apex::start(timer);
        timer_stack.push(timer);
        correlation_map_mutex.lock();
        correlation_map[data->correlation_id] = timer;
        correlation_map_mutex.unlock();
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            apex::stop(timer);
            timer_stack.pop();
        }
    }
    return;
}

/* This is the "OpenMP" API - lots of events if interested. */
void handle_roc_hsa(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    APEX_UNUSED(domain);
    APEX_UNUSED(arg);
    static bool once = run_once();
    APEX_UNUSED(once);
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    const hsa_api_data_t* data = (const hsa_api_data_t*)(callback_data);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        auto timer = apex::new_task(
			roctracer_op_string(ACTIVITY_DOMAIN_HSA_API, cid, 0));
        apex::start(timer);
        timer_stack.push(timer);
        correlation_map_mutex.lock();
        correlation_map[data->correlation_id] = timer;
        correlation_map_mutex.unlock();
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            apex::stop(timer);
            timer_stack.pop();
        }
    }
    return;
}

/* Handle counters from synchronous callbacks */
void store_sync_counter_data(const char * name, const std::string& context,
    double value, bool threaded) {
    if (name == nullptr) {
        std::stringstream ss;
        ss << "GPU: " << context;
        apex::sample_value(ss.str(), value, threaded);
    } else {
        std::stringstream ss;
        ss << "GPU: " << name << ": " << context;
        apex::sample_value(ss.str(), value, threaded);
    }
}

bool getBytesIfMalloc(uint32_t cid, const hip_api_data_t* data,
    std::string context, bool isEnter) {
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
        switch (cid) {
            /*
            case HIP_API_ID_hipMallocPitch: {
                ptr = *data->args.hipMallocPitch.ptr;
                bytes = data->args.hipMallocPitch.width * data->args.hipMallocPitch.height;
                break;
            }
            */
            case HIP_API_ID_hipMalloc: {
                ptr = *data->args.hipMalloc.ptr;
                bytes = data->args.hipMalloc.size;
                break;
            }
            /*
            case HIP_API_ID_hipMalloc3DArray: {
                ptr = ?
                bytes = ?
                break;
            }
            */
            case HIP_API_ID_hipMallocHost: {
                ptr = *data->args.hipMallocHost.ptr;
                bytes = data->args.hipMallocHost.size;
                onHost = true;
                break;
            }
            /*
            case HIP_API_ID_hipMallocArray: {
                ptr = ?
                bytes = ?
                break;
            }
            */
            /*
            case HIP_API_ID_hipMallocMappedArray: {
                ptr = ?
                bytes = ?
                break;
            }
            */
            case HIP_API_ID_hipHostMalloc: {
                ptr = *data->args.hipHostMalloc.ptr;
                bytes = data->args.hipHostMalloc.size;
                onHost = true;
                break;
            }
            case HIP_API_ID_hipMallocManaged: {
                ptr = *data->args.hipMallocManaged.dev_ptr;
                bytes = data->args.hipMallocManaged.size;
                managed = true;
                break;
            }
            /*
            case HIP_API_ID_hipMalloc3D: {
                ptr = ?
                bytes = ?
                break;
            }
            */
            case HIP_API_ID_hipExtMallocWithFlags: {
                ptr = *data->args.hipExtMallocWithFlags.ptr;
                bytes = data->args.hipExtMallocWithFlags.sizeBytes;
                break;
            }
            case HIP_API_ID_hipFreeHost: {
                ptr = data->args.hipFreeHost.ptr;
                free = true;
                onHost = true;
                break;
            }
            case HIP_API_ID_hipFreeArray: {
                ptr = data->args.hipFreeArray.array;
                free = true;
                break;
            }
            /*
            case HIP_API_ID_hipFreeMipmappedArray: {
                ptr = data->args.hipFreeMipmappedArray.mimappedArray;
                free = true;
                break;
            }
            */
            case HIP_API_ID_hipFree: {
                ptr = data->args.hipFree.ptr;
                free = true;
                break;
            }
            case HIP_API_ID_hipHostFree: {
                ptr = data->args.hipHostFree.ptr;
                free = true;
                onHost = true;
                break;
            }
            default: {
                return false;
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
            store_sync_counter_data("Host Bytes Freed", context, value, true);
            hostTotalAllocated.fetch_sub(bytes, std::memory_order_relaxed);
            value = (double)(hostTotalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Host", value, false);
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
            store_sync_counter_data("Bytes Freed", context, value, true);
            totalAllocated.fetch_sub(value, std::memory_order_relaxed);
            value = (double)(totalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Device", value, false);
        }
    // If we are in the exit of a function, and we are allocating memory,
    // then update and record the bytes allocated
    } else if (!free && !isEnter) {
        if (bytes == 0) return false;
        double value = (double)(bytes);
        //std::cout << "Allocating " << value << " bytes at " << ptr << std::endl;
        if (onHost) {
            store_sync_counter_data("Bytes Allocated", context, value, true);
            hostMapMutex.lock();
            hostMemoryMap[ptr] = value;
            hostMapMutex.unlock();
            hostTotalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            value = (double)(hostTotalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Host", value, false);
            return true;
        } else {
            if (managed) {
                store_sync_counter_data("Bytes Allocated (Managed)", context, value, true);
            } else {
                store_sync_counter_data("Bytes Allocated", context, value, true);
            }
            mapMutex.lock();
            memoryMap[ptr] = value;
            mapMutex.unlock();
            totalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            value = (double)(totalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Device", value, false);
        }
    }
    return true;
}

/* The HIP callback API.  For these events, we have to check whether it's
 * the entry or exit event, and act accordingly.
 */
void handle_hip(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    APEX_UNUSED(domain);
    APEX_UNUSED(arg);
    static bool once = run_once();
    APEX_UNUSED(once);
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
    std::string context{roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0)};
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        auto timer = apex::new_task(context);
        apex::start(timer);
        timer_stack.push(timer);
        correlation_map_mutex.lock();
        correlation_map[data->correlation_id] = timer;
        correlation_map_mutex.unlock();

        switch (cid) {
            case HIP_API_ID_hipMallocPitch:
            case HIP_API_ID_hipMalloc:
            case HIP_API_ID_hipMalloc3DArray:
            case HIP_API_ID_hipMallocHost:
            case HIP_API_ID_hipMallocArray:
            case HIP_API_ID_hipMallocMipmappedArray:
            case HIP_API_ID_hipHostMalloc:
            case HIP_API_ID_hipMallocManaged:
            case HIP_API_ID_hipMalloc3D:
            case HIP_API_ID_hipExtMallocWithFlags:
            case HIP_API_ID_hipFreeHost:
            case HIP_API_ID_hipFreeArray:
            case HIP_API_ID_hipFreeMipmappedArray:
            case HIP_API_ID_hipFree:
            case HIP_API_ID_hipHostFree:
                getBytesIfMalloc(cid, data, context, true);
                break;
            case HIP_API_ID_hipLaunchKernel:
                correlation_map_mutex.lock();
                correlation_kernel_map[data->correlation_id] = (const void*)data->args.hipLaunchKernel.function_address;
                correlation_map_mutex.unlock();
                break;
            case HIP_API_ID_hipModuleLaunchKernel:
                correlation_map_mutex.lock();
                correlation_kernel_map[data->correlation_id] = (const void*)data->args.hipModuleLaunchKernel.f;
                correlation_map_mutex.unlock();
                break;
            case HIP_API_ID_hipHccModuleLaunchKernel:
                correlation_map_mutex.lock();
                correlation_kernel_map[data->correlation_id] = (const void*)data->args.hipHccModuleLaunchKernel.f;
                correlation_map_mutex.unlock();
                break;
            case HIP_API_ID_hipExtModuleLaunchKernel:
                correlation_map_mutex.lock();
                correlation_kernel_map[data->correlation_id] = (const void*)data->args.hipExtModuleLaunchKernel.f;
                correlation_map_mutex.unlock();
                break;
            case HIP_API_ID_hipExtLaunchKernel:
                correlation_map_mutex.lock();
                correlation_kernel_map[data->correlation_id] = (const void*)data->args.hipExtLaunchKernel.function_address;
                correlation_map_mutex.unlock();
                break;
            default:
                break;
        }
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            apex::stop(timer);
            timer_stack.pop();
        }

        switch (cid) {
            case HIP_API_ID_hipMallocPitch:
            case HIP_API_ID_hipMalloc:
            case HIP_API_ID_hipMalloc3DArray:
            case HIP_API_ID_hipMallocHost:
            case HIP_API_ID_hipMallocArray:
            case HIP_API_ID_hipMallocMipmappedArray:
            case HIP_API_ID_hipHostMalloc:
            case HIP_API_ID_hipMallocManaged:
            case HIP_API_ID_hipMalloc3D:
            case HIP_API_ID_hipExtMallocWithFlags:
            case HIP_API_ID_hipFreeHost:
            case HIP_API_ID_hipFreeArray:
            case HIP_API_ID_hipFreeMipmappedArray:
            case HIP_API_ID_hipFree:
            case HIP_API_ID_hipHostFree:
                getBytesIfMalloc(cid, data, context, false);
                break;
            default:
                break;
        }
    }
}

void store_profiler_data(const std::string &name, uint32_t correlationId,
        uint64_t start, uint64_t end, apex::hip_thread_node &node,
        bool otf2_trace = true) {
    apex::in_apex prevent_deadlocks;
    // Get the singleton APEX instance
    static apex::apex* instance = apex::apex::instance();
    // get the parent GUID, then erase the correlation from the map
    std::shared_ptr<apex::task_wrapper> parent = nullptr;
    if (correlationId > 0) {
        correlation_map_mutex.lock();
        parent = correlation_map[correlationId];
        correlation_map.erase(correlationId);
        correlation_map_mutex.unlock();
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
    if (apex::apex_options::use_trace_event()) {
        apex::trace_event_listener * tel =
            (apex::trace_event_listener*)instance->the_trace_event_listener;
        tel->on_async_event(node, prof);
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

/* Handle counters from asynchronous activity */
void store_counter_data(const char * name, const std::string& ctx,
    uint64_t end, double value, apex::hip_thread_node &node) {
    apex::in_apex prevent_deadlocks;
    std::stringstream ss;
    if (name == nullptr) {
        ss << "GPU: " << ctx;
    } else {
        ss << "GPU: " << name << " " << ctx;
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
    uint64_t end, size_t value, apex::hip_thread_node &node) {
    store_counter_data(name, ctx, end, (double)(value), node);
}

void process_hip_record(const roctracer_record_t* record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    switch(record->op) {
        case HIP_OP_ID_DISPATCH: {
            correlation_map_mutex.lock();
            const void* f = correlation_kernel_map[record->correlation_id];
            correlation_kernel_map.erase(record->correlation_id);
            correlation_map_mutex.unlock();
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_KERNEL);
            std::stringstream ss;
            ss << "UNRESOLVED ADDR " << std::hex << f ;
   	        store_profiler_data(ss.str(), record->correlation_id, record->begin_ns,
                record->end_ns, node);
            break;
        }
        case HIP_OP_ID_COPY: {
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_MEMORY);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, node);
            store_counter_data(name, "Bytes", record->end_ns,
                record->bytes, node);
            break;
        }
        case HIP_OP_ID_BARRIER: {
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_SYNCHRONIZE);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, node);
            break;
        }
        case HIP_OP_ID_NUMBER:
        default: {
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_OTHER);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, node);
            break;
        }
    }
}

// Activity tracing callback
void activity_callback(const char* begin, const char* end, void* arg) {
    APEX_UNUSED(arg);
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

    while (record < end_record) {
        // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
        if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
            process_hip_record(record);
        } else {
            fprintf(stderr, "Bad domain %d\n\n", record->domain);
            abort();
        }

        ROCTRACER_CALL(roctracer_next_record(record, &record));
    }
}

// Init tracing routine
void init_tracing() {
    printf("# INIT #############################\n");
    // roctracer properties
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
    // Allocating tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    properties.buffer_callback_fun = activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));

    // Enable HIP API callbacks
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, handle_hip, NULL));
    // Enable HIP HSA (OpenMP) callbacks
    //ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, handle_roc_hsa, NULL));
    // Enable KFD API tracing
    if (apex::apex_options::use_hip_kfd_api()) {
        ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, handle_roc_kfd, NULL));
    }
    // Enable rocTX
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, handle_roctx, NULL));

    // Enable HIP activity tracing
    //ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    //ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
    // Enable PC sampling
    //ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
    roctracer_start();

}

namespace apex {
    // Stop tracing routine
    void flush_hip_trace() {
        ROCTRACER_CALL(roctracer_flush_activity());
    }

    void stop_hip_trace() {
        roctracer_stop();
        /* CAllbacks */
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
        if (apex_options::use_hip_kfd_api()) {
            ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
        }
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));

        /* Activity */
        //ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
        // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_EXT_API));
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
        ROCTRACER_CALL(roctracer_flush_activity());
        printf("# STOP  #############################\n");
    }
} // namespace apex

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
