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

#ifdef __cplusplus
#include <cstdlib>
using namespace std;
#else
#include <stdlib.h>
#endif

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#include "apex_api.hpp"
#include <stack>
#include <mutex>
#include <map>

static thread_local const size_t msg_size = 512;
static thread_local char* msg_buf = NULL;
static thread_local char* message = NULL;

#if 0
void SPRINT(const char* fmt, ...) {
    if (msg_buf == NULL) {
        msg_buf = (char*) calloc(msg_size, 1);
        message = msg_buf;
    }

    va_list args;
    va_start(args, fmt);
    message += vsnprintf(message, msg_size - (message - msg_buf), fmt, args);
    va_end(args);
}
void SFLUSH() {
    if (msg_buf == NULL) abort();
    message = msg_buf;
    msg_buf[msg_size - 1] = 0;
    fprintf(stdout, "%s", msg_buf);
    fflush(stdout);
}
#else
void SPRINT(const char* fmt, ...) { }
void SFLUSH() { }
#endif

static void __attribute__((constructor)) init_tracing(void);

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
#define ROCTRACER_CALL(call)                                                                       \
    do {                                                                                             \
        int err = call;                                                                                \
        if (err != 0) {                                                                                \
            fprintf(stderr, "%s\n", roctracer_error_string());                                                    \
            abort();                                                                                     \
        }                                                                                              \
    } while (0)

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }

/* This is like the CUDA NVTX API.  User-added instrumentation for
 * ranges that can be pushed/popped on a stack (and common to a thread
 * of execution) or started/stopped (and can be started by one thread
 * and stopped by another).
 */
void handle_roctx(uint32_t cid, const void* callback_data, void* arg) {
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

/* This is the "low level" API - lots of events if interested. */
void handle_roc_kfd(uint32_t cid, const void* callback_data, void* arg) {
    const kfd_api_data_t* data = (const kfd_api_data_t*)(callback_data);
    SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)>\n",
            roctracer_op_string(ACTIVITY_DOMAIN_KFD_API, cid, 0),
            cid,
            data->correlation_id,
            (data->phase == ACTIVITY_API_PHASE_ENTER) ?
            "on-enter" : "on-exit", GetPid(), GetTid());
    return;
}

/* The map that holds correlation IDs and matches them to GUIDs */
std::unordered_map<uint32_t, std::shared_ptr<apex::task_wrapper>> correlation_map;
std::mutex correlation_map_mutex;

/* The HIP callback API.  For these events, we have to check whether it's
 * the entry or exit event, and act accordingly.
 */
void handle_hip(uint32_t cid, const void* callback_data, void* arg) {
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
    SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)> ",
            roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
            cid,
            data->correlation_id,
            (data->phase == ACTIVITY_API_PHASE_ENTER) ?
            "on-enter" : "on-exit", GetPid(), GetTid());
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        auto timer = apex::new_task(
			roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0));
        apex::start(timer);
        timer_stack.push(timer);
        correlation_map_mutex.lock();
        correlation_map[data->correlation_id] = timer;
        correlation_map_mutex.unlock();

        switch (cid) {
            case HIP_API_ID_hipMemcpy:
                SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)",
                        data->args.hipMemcpy.dst,
                        data->args.hipMemcpy.src,
                        (uint32_t)(data->args.hipMemcpy.sizeBytes),
                        (uint32_t)(data->args.hipMemcpy.kind));
                break;
            case HIP_API_ID_hipMalloc:
                SPRINT("ptr(%p) size(0x%x)",
                        data->args.hipMalloc.ptr,
                        (uint32_t)(data->args.hipMalloc.size));
                break;
            case HIP_API_ID_hipFree:
                SPRINT("ptr(%p)", data->args.hipFree.ptr);
                break;
            case HIP_API_ID_hipModuleLaunchKernel:
                SPRINT("kernel(\"%s\") stream(%p)",
                        hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
                        data->args.hipModuleLaunchKernel.stream);
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
            case HIP_API_ID_hipMalloc:
                SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
                break;
            default:
                break;
        }
    }
    SPRINT("\n");
    SFLUSH();
}

// Runtime API callback function
void api_callback(uint32_t domain, uint32_t cid,
        const void* callback_data, void* arg) {
    (void)arg;

    if (domain == ACTIVITY_DOMAIN_ROCTX) {
        handle_roctx(cid, callback_data, arg);
        return;
    }
    if (domain == ACTIVITY_DOMAIN_KFD_API) {
        handle_roc_kfd(cid, callback_data, arg);
        return;
    }
    /* Everything else is HIP API */
    handle_hip(cid, callback_data, arg);
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

    SPRINT("\tActivity records:\n");
    while (record < end_record) {
        const char * name = roctracer_op_string(record->domain, record->op, record->kind);
        SPRINT("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
                name,
                record->correlation_id,
                record->begin_ns,
                record->end_ns);
        if ((record->domain == ACTIVITY_DOMAIN_HIP_API) || (record->domain == ACTIVITY_DOMAIN_KFD_API)) {
            SPRINT(" process_id(%u) thread_id(%u)",
                    record->process_id,
                    record->thread_id);
        } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
            SPRINT(" device_id(%d) queue_id(%lu)",
                    record->device_id,
                    record->queue_id);
            if (record->op == HIP_OP_ID_COPY) SPRINT(" bytes(0x%zx)", record->bytes);
        } else if (record->domain == ACTIVITY_DOMAIN_HSA_OPS) {
            SPRINT(" se(%u) cycle(%lu) pc(%lx)",
                    record->pc_sample.se,
                    record->pc_sample.cycle,
                    record->pc_sample.pc);
        } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
            SPRINT(" external_id(%lu)", record->external_id);
        } else {
            fprintf(stderr, "Bad domain %d\n\n", record->domain);
            abort();
        }
        SPRINT("\n");
        SFLUSH();

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
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL));
    // Enable HIP activity tracing
    //#if HIP_API_ACTIVITY_ON
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    //#endif
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    // Enable PC sampling
    ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
    // Enable KFD API tracing
    //ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, api_callback, NULL));
    // Enable rocTX
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL));
    roctracer_start();
}

namespace apex {
    // Stop tracing routine
    void flush_hip_trace() {
        ROCTRACER_CALL(roctracer_flush_activity());
    }

    void stop_hip_trace() {
        roctracer_stop();
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
        //#if HIP_API_ACTIVITY_ON
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
        //#endif
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
        ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
        ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
        ROCTRACER_CALL(roctracer_flush_activity());
        printf("# STOP  #############################\n");
    }
} // namespace apex

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
