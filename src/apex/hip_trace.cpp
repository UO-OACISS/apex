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

#define APEX_WITH_HSA

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#include "apex_api.hpp"
#include "apex.hpp"
#ifdef APEX_HAVE_BFD
#include "address_resolution.hpp"
#endif
#include "async_thread_node.hpp"
#if defined(APEX_WITH_PERFETTO)
#include "perfetto_listener.hpp"
#endif
#include "trace_event_listener.hpp"
#ifdef APEX_HAVE_OTF2
#include "otf2_listener.hpp"
#endif
#include <stack>
#include <mutex>
#include <map>

#include "global_constructor_destructor.h"
/* NO!  Now called directly from APEX initialization, to make sure that PAPI
 * or Rocprofiler are initiallized before calling hsa_init() */
// DEFINE_CONSTRUCTOR(init_hip_tracing);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
// Required by the hip header
#if !defined(__HIP_PLATFORM_AMD__)
#define __HIP_PLATFORM_AMD__
#endif
#if !defined(AMD_INTERNAL_BUILD)
#define AMD_INTERNAL_BUILD
#endif
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#if defined(APEX_WITH_HSA)
// This rquires the -DAMD_INTERNAL_BUILD compiler flag
#include <roctracer_hsa.h>
#endif
#if defined(APEX_HAVE_ROCTRACER_KFD)
#include <roctracer_kfd.h>
#endif
#include <roctracer_roctx.h>

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <inttypes.h>

#include "memory_wrapper.hpp"

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL_CHECK(call)                                    \
    do {                                                        \
        int err = call;                                         \
        if (err != 0) {                                         \
            fprintf(stderr, "%s\n", roctracer_error_string());  \
            abort();                                            \
        }                                                       \
    } while (0)

/* This class is necessary so we can clean up before our globals are destroyed at exit */
class Globals{
private:
    Globals() : deltaTimestamp(0), bad_stacks(false) {} // Disallow instantiation outside of the class.
    // Timestamp at trace initialization time. Used to normalized other
    // timestamps
    int64_t deltaTimestamp;
    bool bad_stacks;
    std::mutex correlation_map_mutex;
    std::unordered_map<uint32_t, std::shared_ptr<apex::task_wrapper>> timer_map;
    std::unordered_map<uint32_t, std::string> name_map;
    std::unordered_map<uint32_t, apex::async_event_data> data_map;
public:
    Globals(const Globals&) = delete;
    Globals& operator=(const Globals &) = delete;
    Globals(Globals &&) = delete;
    Globals & operator=(Globals &&) = delete;
    // if our globals are destroyed before APEX can finalize, then finalize now! */
    ~Globals() {
        if (bad_stacks) {
            std::cerr << "Warning! APEX detected more timer stops than starts.\n";
            std::cerr << "\tPlease confirm that you have correctly matched roctx push/pops.";
        }
        apex::finalize();
    }

    static auto& instance(){
        static Globals test;
        return test;
    }
    static int64_t& delta() {
        return instance().deltaTimestamp;
    }

    void lock() { correlation_map_mutex.lock(); }
    void unlock() { correlation_map_mutex.unlock(); }

    static void insert_timer(uint32_t id, std::shared_ptr<apex::task_wrapper> timer) {
        instance().correlation_map_mutex.lock();
        instance().timer_map[id] = timer;
        instance().correlation_map_mutex.unlock();
    }

    static std::shared_ptr<apex::task_wrapper> find_timer(uint32_t id) {
        std::shared_ptr<apex::task_wrapper> timer = nullptr;
        Globals& g = instance();
        g.lock();
        auto iter = g.timer_map.find(id);
        if (iter != g.timer_map.end()) {
            timer = iter->second;
        }
        g.timer_map.erase(id);
        g.unlock();
        return timer;
    }

    static void report_bad_stacks() {
        Globals& g = instance();
        g.bad_stacks = true;
    }

    static void insert_name(uint32_t id, std::string& name) {
        instance().correlation_map_mutex.lock();
        instance().name_map[id] = name;
        instance().correlation_map_mutex.unlock();
    }

    static std::string find_name(uint32_t id) {
        Globals& g = instance();
        std::string name;
        g.lock();
        auto iter = g.name_map.find(id);
        if (iter != g.name_map.end()) {
            name = iter->second;
            g.name_map.erase(id);
        } else {
            name = "unknown";
        }
        g.unlock();
        return name;
    }

    static void insert_data(uint32_t id, apex::async_event_data& data) {
        instance().correlation_map_mutex.lock();
        instance().data_map[id] = data;
        instance().correlation_map_mutex.unlock();
    }

    static apex::async_event_data find_data(uint32_t id) {
        apex::async_event_data as_data;
        Globals& g = instance();
        g.lock();
        auto iter = g.data_map.find(id);
        if (iter != g.data_map.end()) {
            as_data = iter->second;
        }
        g.data_map.erase(id);
        g.unlock();
        return as_data;
    }
};

constexpr char devicestub[] = "__device_stub__";

/* Needed to prevent re-entry when profiling with kfd/hsa */
class handler_lock {
public:
    bool mine;
    handler_lock() : mine(false) {
        // is the callstack on this thread aleady in a handler?
        // if not, you get it.
        if (!handling()) {
            mine = true;
            handling() = true;
        }
    }
    // if I have the lock, release it.
    ~handler_lock() { if (mine) handling() = false; }
    // put this in a function, so it can be initialized correctly by some compilers
    bool& handling() {
        static APEX_NATIVE_TLS bool handling{false};
        return handling;
    }
};

/* SHOULD only effectively get called from init */
bool run_once() {
    static bool once{false};
    if (once) { return once; }
    constexpr int attempts = 10;
    uint64_t startTimestampGPU[attempts] = {0};
    uint64_t startTimestampCPU[attempts] = {0};
    // NOTE: This code isn't really useful.  It would be if we were
    // using CPU timestamps for the host activity, but we can't seem
    // to sync the clocks right, so *ALL* timestamps are taken on the
    // GPU for now.
    // synchronize timestamps
    // this is a common algorithm, see otf2_listener_mpi.cpp for synchronizing across ranks
    for (int i = 0 ; i < attempts ; i++) {
        roctracer_get_timestamp(&(startTimestampGPU[i]));
        startTimestampCPU[i] = apex::profiler::now_ns();
    }
    // assume the GPU clock is less than the CPU clock
    int64_t latency = (int64_t)(startTimestampCPU[0]) - (int64_t)(startTimestampGPU[0]);
    int my_min{0};
    for (int i = 1 ; i < attempts ; i++) {
        int64_t next = (int64_t)(startTimestampCPU[i]) - (int64_t)(startTimestampGPU[i]);
        if (std::abs(next) < std::abs(latency)) {
            latency = next;
            my_min = i;
        }
    }

    // assume CPU timestamp is greater than GPU
    Globals::delta() = ((int64_t)(startTimestampCPU[my_min]) - (int64_t)(startTimestampGPU[my_min]));
    //printf("HIP timestamp:      %" PRIu64 "\n", startTimestampGPU[my_min]);
    //printf("CPU timestamp:      %" PRIu64 "\n", startTimestampCPU[my_min]);
    //printf("HIP delta timestamp: %" PRId64 "\n", Globals::delta());
    once = true;
    return once;
}

/* This is like the CUDA NVTX API.  User-added instrumentation for
 * ranges that can be pushed/popped on a stack (and common to a thread
 * of execution) or started/stopped (and can be started by one thread
 * and stopped by another).
 */
void handle_roctx(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }
    // prevent re-entry
    handler_lock hl;
    if (!hl.mine) { return; }
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
                ss /* << "roctx: " */ << data->args.message;
                timer_stack.push(apex::start(ss.str()));
                break;
            }
        case ROCTX_API_ID_roctxRangePop:
            {
                if (!timer_stack.empty()) {
                    apex::stop(timer_stack.top());
                    timer_stack.pop();
                } else { Globals::report_bad_stacks(); }
                break;
            }
        case ROCTX_API_ID_roctxRangeStartA:
            {
                std::stringstream ss;
                ss /* << "roctx: " */ << data->args.message;
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

#if defined(APEX_HAVE_ROCTRACER_KFD)
/* This is the "low level" API - lots of events if interested. */
void handle_roc_kfd(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }
    // ignore timestamp requests
    if (cid == KFD_API_ID_hsaKmtGetClockCounters) return;
    // prevent re-entry
    handler_lock hl;
    if (!hl.mine) { return; }
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
        Globals::insert_timer(data->correlation_id, timer);
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            apex::stop(timer);
            timer_stack.pop();
        } else { Globals::report_bad_stacks(); }
    }
    return;
}
#endif

/* This is the "OpenMP" API - lots of events if interested. */
#if defined(APEX_WITH_HSA)
void handle_roc_hsa(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }
    // ignore timestamp requests
#if defined(APEX_HAVE_ROCTRACER_KFD)
    if (cid == KFD_API_ID_hsaKmtGetClockCounters) return;
#endif
    // prevent re-entry
    handler_lock hl;
    if (!hl.mine) { return; }
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
        Globals::insert_timer(data->correlation_id, timer);
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            apex::stop(timer);
            timer_stack.pop();
        } else { Globals::report_bad_stacks(); }
    }
    return;
}
#endif

/* Handle counters from synchronous callbacks */
void store_sync_counter_data(const char * name, const std::string& context,
    double value, bool threaded) {
    if (name == nullptr) {
        std::stringstream ss;
        ss << "GPU: " << context;
        apex::sample_value(ss.str(), value, threaded);
    } else {
        std::stringstream ss;
        ss << "GPU: " << name;
        if (context.size() > 0 &&
            !apex::apex_options::use_perfetto() &&
            !apex::apex_options::use_trace_event() &&
            !apex::apex_options::use_otf2()) {
            ss << ": " << context;
        }
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
            store_sync_counter_data("Bytes Freed", context, value, true);
            totalAllocated.fetch_sub(value, std::memory_order_relaxed);
            value = (double)(totalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Device", value, false);
            apex::recordFree(ptr, false);
        }
    // If we are in the exit of a function, and we are allocating memory,
    // then update and record the bytes allocated
    } else if (!free && !isEnter) {
        if (bytes == 0) return false;
        double value = (double)(bytes);
        //std::cout << "Allocating " << value << " bytes at " << ptr << std::endl;
        if (onHost) {
            store_sync_counter_data("Host Bytes Allocated", context, value, true);
            hostMapMutex.lock();
            hostMemoryMap[ptr] = value;
            hostMapMutex.unlock();
            hostTotalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            value = (double)(hostTotalAllocated);
            store_sync_counter_data(nullptr, "Total Bytes Occupied on Host", value, false);
            apex::recordAlloc(bytes, ptr, apex::GPU_HOST_MALLOC);
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
            apex::recordAlloc(bytes, ptr, apex::GPU_DEVICE_MALLOC, false);
        }
    }
    return true;
}

std::string lookup_kernel_name_ptr(const void* address, hipStream_t stream_id) {
    // no need to lock, because we are locked before entering this function
    static std::map<const void*, std::string> the_map;
    auto entry = the_map.find(address);
    std::string tmp;
    if (entry != the_map.end()) {
        tmp = entry->second;
    } else {
#ifdef APEX_HAVE_BFD
        // look it up using the right method
        if (apex::apex_options::use_source_location()) {
            tmp = *(apex::lookup_address((uintptr_t)address, true));
            tmp = apex::demangle(tmp);
            // clean up the kernel name (strip '__device_stub__')
            size_t pos = tmp.find(devicestub);
            if (pos != std::string::npos) {
                tmp.erase(pos, strlen(devicestub));
            }
        } else {
            tmp = std::string(hipKernelNameRefByPtr(address, stream_id));
            tmp = apex::demangle(tmp);
        }
#else
        tmp = std::string(hipKernelNameRefByPtr(address, stream_id));
        tmp = apex::demangle(tmp);
#endif
        // add it to the map
        the_map.insert(std::make_pair(address, tmp));
    }
    return tmp;
}

std::string lookup_kernel_name(const hipFunction_t f) {
    // no need to lock, because we are locked before entering this function
    static std::map<const hipFunction_t, std::string> the_map;
    auto entry = the_map.find(f);
    std::string tmp;
    if (entry != the_map.end()) {
        tmp = entry->second;
    } else {
#ifdef APEX_HAVE_BFD
        if (apex::apex_options::use_source_location()) {
            tmp = *(apex::lookup_address((uintptr_t)f, true));
            tmp = apex::demangle(tmp);
            // clean up the kernel name (strip '__device_stub__')
            size_t pos = tmp.find(devicestub);
            if (pos != std::string::npos) {
                tmp.erase(pos, strlen(devicestub));
            }
        } else {
            tmp = hipKernelNameRef(f);
            tmp = apex::demangle(tmp);
        }
#else
        tmp = hipKernelNameRef(f);
        tmp = apex::demangle(tmp);
#endif
        // add it to the map
        the_map.insert(std::make_pair(f, tmp));
    }
    return tmp;
}

/* The HIP callback API.  For these events, we have to check whether it's
 * the entry or exit event, and act accordingly.
 */
void handle_hip(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }
    // prevent re-entry
    handler_lock hl;
    if (!hl.mine) { return; }
    APEX_UNUSED(domain);
    APEX_UNUSED(arg);
    static bool once = run_once();
    APEX_UNUSED(once);
    /* Check for a couple of useless callbacks, we don't need to track them */
    if (cid == HIP_API_ID___hipPushCallConfiguration ||
        cid == HIP_API_ID___hipPopCallConfiguration) {
        return;
    }
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
    std::string context{roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0)};
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        auto timer = apex::new_task(context);
        apex::start(timer);
        timer_stack.push(timer);
        Globals::insert_timer(data->correlation_id, timer);

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
            {
                std::string name{lookup_kernel_name_ptr(
                        data->args.hipLaunchKernel.function_address,
                        data->args.hipLaunchKernel.stream)};
                Globals::insert_name(data->correlation_id, name);
                if (apex::apex_options::use_hip_kernel_details()) {
                    store_sync_counter_data("numBlocks.X", name,
                        data->args.hipLaunchKernel.numBlocks.x, true);
                    store_sync_counter_data("numBlocks.Y", name,
                        data->args.hipLaunchKernel.numBlocks.y, true);
                    store_sync_counter_data("numBlocks.Z", name,
                        data->args.hipLaunchKernel.numBlocks.z, true);
                    store_sync_counter_data("dimBlocks.X", name,
                        data->args.hipLaunchKernel.dimBlocks.x, true);
                    store_sync_counter_data("dimBlocks.Y", name,
                        data->args.hipLaunchKernel.dimBlocks.y, true);
                    store_sync_counter_data("dimBlocks.Z", name,
                        data->args.hipLaunchKernel.dimBlocks.z, true);
                    store_sync_counter_data("sharedMemBytes", name,
                        data->args.hipLaunchKernel.sharedMemBytes, true);
                }
                break;
            }
            case HIP_API_ID_hipModuleLaunchKernel:
            {
                std::string name {lookup_kernel_name(data->args.hipModuleLaunchKernel.f)};
                Globals::insert_name(data->correlation_id, name);
                if (apex::apex_options::use_hip_kernel_details()) {
                    store_sync_counter_data("blockDim.X", name,
                        data->args.hipModuleLaunchKernel.blockDimX, true);
                    store_sync_counter_data("blockDim.Y", name,
                        data->args.hipModuleLaunchKernel.blockDimY, true);
                    store_sync_counter_data("blockDim.Z", name,
                        data->args.hipModuleLaunchKernel.blockDimZ, true);
                    store_sync_counter_data("gridDim.X", name,
                        data->args.hipModuleLaunchKernel.gridDimX, true);
                    store_sync_counter_data("gridDim.Y", name,
                        data->args.hipModuleLaunchKernel.gridDimY, true);
                    store_sync_counter_data("gridDim.Z", name,
                        data->args.hipModuleLaunchKernel.gridDimZ, true);
                    store_sync_counter_data("sharedMemBytes", name,
                        data->args.hipModuleLaunchKernel.sharedMemBytes, true);
                }
                break;
            }
            case HIP_API_ID_hipHccModuleLaunchKernel:
            {
                std::string name =
                    lookup_kernel_name(data->args.hipHccModuleLaunchKernel.f);
                Globals::insert_name(data->correlation_id, name);
                if (apex::apex_options::use_hip_kernel_details()) {
                    store_sync_counter_data("blockDim.X", name,
                        data->args.hipHccModuleLaunchKernel.blockDimX, true);
                    store_sync_counter_data("blockDim.Y", name,
                        data->args.hipHccModuleLaunchKernel.blockDimY, true);
                    store_sync_counter_data("blockDim.Z", name,
                        data->args.hipHccModuleLaunchKernel.blockDimZ, true);
                    store_sync_counter_data("globalWorkSize.X", name,
                        data->args.hipHccModuleLaunchKernel.globalWorkSizeX, true);
                    store_sync_counter_data("globalWorkSize.Y", name,
                        data->args.hipHccModuleLaunchKernel.globalWorkSizeY, true);
                    store_sync_counter_data("globalWorkSize.Z", name,
                        data->args.hipHccModuleLaunchKernel.globalWorkSizeZ, true);
                    store_sync_counter_data("sharedMemBytes", name,
                        data->args.hipHccModuleLaunchKernel.sharedMemBytes, true);
                }
                break;
            }
            case HIP_API_ID_hipExtModuleLaunchKernel:
            {
                std::string name =
                    lookup_kernel_name(data->args.hipExtModuleLaunchKernel.f);
                Globals::insert_name(data->correlation_id, name);
                if (apex::apex_options::use_hip_kernel_details()) {
                    store_sync_counter_data("globalWorkSize.X", name,
                        data->args.hipExtModuleLaunchKernel.globalWorkSizeX, true);
                    store_sync_counter_data("globalWorkSize.Y", name,
                        data->args.hipExtModuleLaunchKernel.globalWorkSizeY, true);
                    store_sync_counter_data("globalWorkSize.Z", name,
                        data->args.hipExtModuleLaunchKernel.globalWorkSizeZ, true);
                    store_sync_counter_data("localWorkSize.X", name,
                        data->args.hipExtModuleLaunchKernel.localWorkSizeX, true);
                    store_sync_counter_data("localWorkSize.Y", name,
                        data->args.hipExtModuleLaunchKernel.localWorkSizeY, true);
                    store_sync_counter_data("localWorkSize.Z", name,
                        data->args.hipExtModuleLaunchKernel.localWorkSizeZ, true);
                    store_sync_counter_data("sharedMemBytes", name,
                        data->args.hipExtModuleLaunchKernel.sharedMemBytes, true);
                }
                break;
            }
            case HIP_API_ID_hipExtLaunchKernel:
            {
                std::string name =
                    lookup_kernel_name_ptr(
                        data->args.hipExtLaunchKernel.function_address,
                        data->args.hipExtLaunchKernel.stream);
                Globals::insert_name(data->correlation_id, name);
                if (apex::apex_options::use_hip_kernel_details()) {
                    store_sync_counter_data("numBlocks.X", name,
                        data->args.hipExtLaunchKernel.numBlocks.x, true);
                    store_sync_counter_data("numBlocks.Y", name,
                        data->args.hipExtLaunchKernel.numBlocks.y, true);
                    store_sync_counter_data("numBlocks.Z", name,
                        data->args.hipExtLaunchKernel.numBlocks.z, true);
                    store_sync_counter_data("dimBlocks.X", name,
                        data->args.hipExtLaunchKernel.dimBlocks.x, true);
                    store_sync_counter_data("dimBlocks.Y", name,
                        data->args.hipExtLaunchKernel.dimBlocks.y, true);
                    store_sync_counter_data("dimBlocks.Z", name,
                        data->args.hipExtLaunchKernel.dimBlocks.z, true);
                    store_sync_counter_data("sharedMemBytes", name,
                        data->args.hipExtLaunchKernel.sharedMemBytes, true);
                }
                break;
            }
            default:
                break;
        }
    } else {
        if (!timer_stack.empty()) {
            auto timer = timer_stack.top();
            /* This happens if we have already shut down, after Finalize. */
            if (timer != nullptr) {
                apex::async_event_data as_data(
                    timer->prof->get_start_us(),
                    "OtherFlow", data->correlation_id,
                    apex::thread_instance::get_id(), context);
                as_data.parent_ts_stop = apex::profiler::now_us();
                apex::stop(timer);
                Globals::insert_data(data->correlation_id, as_data);
            }
            timer_stack.pop();
        } else { Globals::report_bad_stacks(); }

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
        uint64_t start, uint64_t end, std::string category, apex::hip_thread_node &node,
        bool reverse_flow = false, bool otf2_trace = true) {
    apex::in_apex prevent_deadlocks;
    // Get the singleton APEX instance
    static apex::apex* instance = apex::apex::instance();
    // get the parent GUID, then erase the correlation from the map
    std::shared_ptr<apex::task_wrapper> parent = nullptr;
    apex::async_event_data as_data;
    if (correlationId > 0) {
        parent = Globals::find_timer(correlationId);
        as_data = Globals::find_data(correlationId);
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
    prof->set_start(start + Globals::delta());
    prof->set_end(end + Globals::delta());
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
        as_data.reverse_flow = reverse_flow;
        tel->on_async_event(node, prof, as_data);
    }
#endif
    if (apex::apex_options::use_trace_event()) {
        apex::trace_event_listener * tel =
            (apex::trace_event_listener*)instance->the_trace_event_listener;
        as_data.cat = category;
        as_data.reverse_flow = reverse_flow;
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
    prof->set_end(end + Globals::delta());
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
    uint64_t end, size_t value, apex::hip_thread_node &node) {
    store_counter_data(name, ctx, end, (double)(value), node);
}

void process_hip_record(const roctracer_record_t* record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    if (!apex::apex_options::use_hip_kernel_details()) {
        if (strncmp(name, "Marker", 6) == 0) {
            // if there's a correlation ID, clear it from the maps
            if (record->correlation_id > 0) {
                Globals::find_timer(record->correlation_id);
                Globals::find_data(record->correlation_id);
            }
            // do nothing, this event is annoying.
            return;
        }
    }
    switch(record->op) {
        case HIP_OP_ID_DISPATCH: {
            std::string name = Globals::find_name(record->correlation_id);
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_KERNEL);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, "ControlFlow", node);
            break;
        }
        case HIP_OP_ID_COPY: {
            /* Handle the "Unknown command type" object. This happens with a burst of activity
               when using managed memory. Ignore the event for now. */
            if (record->domain == 2 && record->kind == 1) {
                break;
            }
            /*
            if (strcmp(name, "Unknown command type") == 0) {
                std::cerr << name << ":" << record->domain << ", COPY, " << record->kind << std::endl;
            } */
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_MEMORY);
            bool reverse_flow = (std::string(name).find("DeviceToHost") != std::string::npos);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, "DataFlow", node, reverse_flow);
            store_counter_data(name, "Bytes", record->end_ns,
                record->bytes, node);
            break;
        }
        case HIP_OP_ID_BARRIER: {
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_SYNCHRONIZE);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, "SyncFlow", node, false, false);
            break;
        }
        case HIP_OP_ID_NUMBER:
        default: {
            apex::hip_thread_node node(record->device_id, record->queue_id, APEX_ASYNC_OTHER);
   	        store_profiler_data(name, record->correlation_id, record->begin_ns,
                record->end_ns, "OtherFlow", node);
            break;
        }
    }
}

// Activity tracing callback
void activity_callback(const char* begin, const char* end, void* arg) {
    APEX_UNUSED(arg);
    // if APEX is disabled, do nothing.
    // if APEX is suspended, do nothing.
    if (apex::apex_options::disable() || apex::apex_options::suspend()) { return; }

    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

    //auto p = apex::scoped_timer("APEX: HIP Activity Buffer Flush");
    //std::cout << "**** FLUSHING ACTIVITY BUFFER ****" << std::endl;
    while (record < end_record) {
        // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
        if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
            process_hip_record(record);
        } else {
            fprintf(stderr, "Unsupported domain %d\n\n", record->domain);
            abort();
        }

        ROCTRACER_CALL_CHECK(roctracer_next_record(record, &record));
    }
}

namespace apex {
// Init tracing routine
void init_hip_tracing() {
    if (!apex_options::use_hip()) { return; }
#if defined(APEX_WITH_HSA)
    /* now safe to initialize hsa - needed to get GPU counter offset */
    hsa_init();
    //std::cout << "HSA Initialized by APEX" << std::endl;
    static bool once = run_once();
    APEX_UNUSED(once);
#endif
    // roctracer properties
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
    // Allocating tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    properties.buffer_callback_fun = activity_callback;
    ROCTRACER_CALL_CHECK(roctracer_open_pool(&properties));

    // Enable HIP API callbacks
    ROCTRACER_CALL_CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, handle_hip, NULL));
    if (apex_options::use_hip_kfd_api()) {
#if defined(APEX_HAVE_ROCTRACER_KFD)
        // Enable KFD API tracing
        ROCTRACER_CALL_CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, handle_roc_kfd, NULL));
#endif
#if defined(APEX_WITH_HSA)
        // Enable HIP HSA (OpenMP?) callbacks
        ROCTRACER_CALL_CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, handle_roc_hsa, NULL));
#endif
    }
    // Enable rocTX
    ROCTRACER_CALL_CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, handle_roctx, NULL));

    // Enable HIP activity tracing
    //ROCTRACER_CALL_CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
    ROCTRACER_CALL_CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    if (apex_options::use_hip_kfd_api()) {
#if defined(APEX_WITH_HSA) // disabled for now to simplify compiling
        ROCTRACER_CALL_CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
#endif
#if defined(APEX_HAVE_ROCTRACER_KFD)
        ROCTRACER_CALL_CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_KFD_API));
#endif
    }
    // Enable PC sampling
    //ROCTRACER_CALL_CHECK(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
    roctracer_start();
    //std::cout << "RocTracer started" << std::endl;
}

    // Stop tracing routine
    void flush_hip_trace() {
        if (!apex_options::use_hip()) { return; }
        ROCTRACER_CALL_CHECK(roctracer_flush_activity());
    }

    void stop_hip_trace() {
        if (!apex_options::use_hip()) { return; }
        roctracer_stop();
        /* CAllbacks */
        ROCTRACER_CALL_CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
        if (apex_options::use_hip_kfd_api()) {
#if defined(APEX_WITH_HSA)
            ROCTRACER_CALL_CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
#endif
#if defined(APEX_HAVE_ROCTRACER_KFD)
            ROCTRACER_CALL_CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
#endif
        }
        ROCTRACER_CALL_CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));

        /* Activity */
        //ROCTRACER_CALL_CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
        // FYI, ACTIVITY_DOMAIN_HIP_OPS = ACTIVITY_DOMAIN_HCC_OPS = ACTIVITY_DOMAIN_HIP_VDI...
        ROCTRACER_CALL_CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
        ROCTRACER_CALL_CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_EXT_API));
        if (apex_options::use_hip_kfd_api()) {
#if defined(APEX_WITH_HSA)
            ROCTRACER_CALL_CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_API));
#endif
#if defined(APEX_HAVE_ROCTRACER_KFD)
            ROCTRACER_CALL_CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_KFD_API));
#endif
        }
        ROCTRACER_CALL_CHECK(roctracer_flush_activity());
#if defined(APEX_WITH_HSA)
        hsa_shut_down();
#endif
    }
} // namespace apex

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
