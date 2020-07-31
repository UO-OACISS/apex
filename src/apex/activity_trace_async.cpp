/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 * using asynchronous handling of activity buffers.
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <stack>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "apex.hpp"
#include "profiler.hpp"
#include "thread_instance.hpp"
#include "apex_options.hpp"
#include "trace_event_listener.hpp"
#include "otf2_listener.hpp"

static void __attribute__((constructor)) initTrace(void);
//static void __attribute__((destructor)) flushTrace(void);

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

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestampGPU{0};
static uint64_t startTimestampCPU{0};
static int64_t deltaTimestamp{0};

/* The callback subscriber */
CUpti_SubscriberHandle subscriber;

/* The buffer count */
std::atomic<uint64_t> num_buffers{0};
std::atomic<uint64_t> num_buffers_processed{0};
bool flushing{false};

/* The map that holds correlation IDs and matches them to GUIDs */
std::unordered_map<uint32_t, std::shared_ptr<apex::task_wrapper>> correlation_map;
std::mutex map_mutex;

bool& get_registered(void) {
    static APEX_NATIVE_TLS bool registered{false};
    return registered;
}

bool register_myself(void) {
    bool& registered = get_registered();
    if (!registered) {
        // make sure APEX knows this is not a worker thread
        apex::thread_instance::instance(false);
        /* make sure the profiler_listener has a queue that this
         * thread can push sampled values to */
        apex::apex::async_thread_setup();
        registered = true;
    }
    return registered;
}

void store_profiler_data(const std::string &name, uint32_t correlationId,
        uint64_t start, uint64_t end, uint32_t device, uint32_t context, uint32_t stream) {
    // Get the singleton APEX instance
    static apex::apex* instance = apex::apex::instance();
    // get the parent GUID, then erase the correlation from the map
    std::shared_ptr<apex::task_wrapper> parent = nullptr;
    if (correlationId > 0) {
        map_mutex.lock();
        parent = correlation_map[correlationId];
        correlation_map.erase(correlationId);
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
    // fake out the profiler_listener
    instance->the_profiler_listener->push_profiler_public(prof);
    if (apex::apex_options::use_trace_event()) {
        apex::trace_event_listener * tel =
            (apex::trace_event_listener*)instance->the_trace_event_listener;
        tel->on_async_event(device, context, stream, prof);
    }
    if (apex::apex_options::use_otf2()) {
        apex::otf2_listener * tol =
            (apex::otf2_listener*)instance->the_otf2_listener;
        tol->on_async_event(device, context, stream, prof);
    }

    // have the listeners handle the end of this task
    instance->complete_task(tt);
}

void store_counter_data(const char * name, const std::string& context,
    uint64_t end, double value, bool force = false) {
    APEX_UNUSED(end);
    if (name == nullptr) {
        apex::sample_value(context, value);
    } else {
        std::stringstream ss;
        ss << name;
        if (apex::apex_options::use_cuda_kernel_details() || force) {
            ss << " <- " << context;
        }
        apex::sample_value(ss.str(), value);
    }
}

void store_counter_data(const char * name, const std::string& context,
    uint64_t end, int32_t value, bool force = false) {
    store_counter_data(name, context, end, (double)(value), force);
}

void store_counter_data(const char * name, const std::string& context,
    uint64_t end, uint32_t value, bool force = false) {
    store_counter_data(name, context, end, (double)(value), force);
}

void store_counter_data(const char * name, const std::string& context,
    uint64_t end, uint64_t value, bool force = false) {
    store_counter_data(name, context, end, (double)(value), force);
}

static const char * getMemcpyKindString(uint8_t kind)
{
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "Memory copy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "Memory copy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "Memory copy HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "Memory copy AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "Memory copy AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "Memory copy AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "Memory copy DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "Memory copy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "Memory copy HtoH";
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
            return "Unified Memory copy HTOD";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            return "Unified Memory copy DTOH";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            return "Unified Memory CPU Page Fault Count";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
            return "Unified Memory GPU Page Fault Groups";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
            return "Unified Memory Trashing";
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
            return "Unified Memory Throttling";
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

#if 0
static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
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
#endif

    static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind)
    {
#if 0
        case CUPTI_ACTIVITY_KIND_DEVICE:
            {
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
                break;
            }
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
            {
                CUpti_ActivityDeviceAttribute *attribute =
                    (CUpti_ActivityDeviceAttribute *)record;
                printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
                        attribute->attribute.cupti, attribute->deviceId,
                        (unsigned long long)attribute->value.vUint64);
                break;
            }
        case CUPTI_ACTIVITY_KIND_CONTEXT:
            {
                CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
                printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
                        context->contextId, context->deviceId,
                        getComputeApiKindString(context->computeApiKind,
                        context->nullStreamId);
                break;
            }
#endif
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
            {
                CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
                std::string name{getMemcpyKindString(memcpy->copyKind)};
                store_profiler_data(name, memcpy->correlationId, memcpy->start,
                        memcpy->end, memcpy->deviceId, memcpy->contextId, 0);
                if (apex::apex_options::use_cuda_counters()) {
                    store_counter_data("GPU: Bytes", name, memcpy->end,
                        memcpy->bytes, true);
                    uint64_t duration = memcpy->end - memcpy->start;
                    // dividing bytes by nanoseconds should give us GB/s
                    double bandwidth = (double)(memcpy->bytes) / (double)(duration);
                    store_counter_data("GPU: Bandwith (GB/s)", name,
                        memcpy->end, bandwidth, true);
                }
                break;
            }
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
            {
                CUpti_ActivityUnifiedMemoryCounter2 *memcpy =
                    (CUpti_ActivityUnifiedMemoryCounter2 *) record;
                std::string name{getUvmCounterKindString(memcpy->counterKind)};
                uint32_t device = getUvmCounterDevice(
                    (CUpti_ActivityUnifiedMemoryCounterKind) memcpy->counterKind,
                    memcpy->srcId, memcpy->dstId);
                if (memcpy->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
                    || memcpy->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH) {
                    store_profiler_data(name, 0, memcpy->start, memcpy->end,
                        device, 0, 0);
                    if (apex::apex_options::use_cuda_counters()) {
                        store_counter_data("GPU: Bytes", name, memcpy->end,
                            memcpy->value, true);
                        uint64_t duration = memcpy->end - memcpy->start;
                        // dividing bytes by nanoseconds should give us GB/s
                        double bandwidth = (double)(memcpy->value) / (double)(duration);
                        store_counter_data("GPU: Bandwith (GB/s)", name,
                            memcpy->end, bandwidth, true);
                    }
                } else if (memcpy->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT) {
                        store_counter_data(nullptr, name, memcpy->start,
                            1);
                } else if (memcpy->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT) {
                        store_counter_data(nullptr, name, memcpy->start,
                            memcpy->value);
                }
                break;
            }
#if 0 // not until CUDA 11
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
            {
                CUpti_ActivityMemcpyPtoP *memcpy = (CUpti_ActivityMemcpyPtoP *) record;
                break;
            }
#endif
        case CUPTI_ACTIVITY_KIND_MEMSET:
            {
                CUpti_ActivityMemset *memset =
                    (CUpti_ActivityMemset *) record;
                static std::string name{"Memset"};
                store_profiler_data(name, memset->correlationId, memset->start,
                    memset->end, memset->deviceId, memset->contextId,
                    memset->streamId);
                break;
            }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
            {
                CUpti_ActivityKernel4 *kernel =
                    (CUpti_ActivityKernel4 *) record;
                std::string tmp = std::string(kernel->name);
                store_profiler_data(tmp, kernel->correlationId, kernel->start,
                    kernel->end, kernel->deviceId, kernel->contextId,
                    kernel->streamId);
                if (apex::apex_options::use_cuda_counters()) {
                    std::string * demangled = apex::demangle(kernel->name);
                    store_counter_data("GPU: Dynamic Shared Memory (B)",
                        *demangled, kernel->end, kernel->dynamicSharedMemory);
                    store_counter_data("GPU: Local Memory Per Thread (B)",
                        *demangled, kernel->end, kernel->localMemoryPerThread);
                    store_counter_data("GPU: Local Memory Total (B)",
                        *demangled, kernel->end, kernel->localMemoryTotal);
                    store_counter_data("GPU: Registers Per Thread",
                        *demangled, kernel->end, kernel->registersPerThread);
                    store_counter_data("GPU: Shared Memory Size (B)",
                        *demangled, kernel->end, kernel->sharedMemoryExecuted);
                    store_counter_data("GPU: Static Shared Memory (B)",
                        *demangled, kernel->end, kernel->staticSharedMemory);
                    delete(demangled);
                }
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
    static bool registered = register_myself();
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

bool getBytesIfMalloc(CUpti_CallbackId id, const void* params, std::string context) {
    size_t bytes = 0;
    switch (id) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
            bytes = ((cudaMalloc_v3020_params_st*)(params))->size;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020: {
            bytes = ((cudaMallocPitch_v3020_params_st*)(params))->width *
                    ((cudaMallocPitch_v3020_params_st*)(params))->height;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020: {
            bytes = ((cudaMallocArray_v3020_params_st*)(params))->width *
                    ((cudaMallocArray_v3020_params_st*)(params))->height;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020: {
            bytes = ((cudaMallocHost_v3020_params_st*)(params))->size;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020: {
            cudaExtent extent = ((cudaMalloc3D_v3020_params_st*)(params))->extent;
            bytes = extent.depth * extent.height * extent.width;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020: {
            cudaExtent extent = ((cudaMalloc3DArray_v3020_params_st*)(params))->extent;
            bytes = extent.depth * extent.height * extent.width;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000: {
            cudaExtent extent = ((cudaMallocMipmappedArray_v5000_params_st*)(params))->extent;
            bytes = extent.depth * extent.height * extent.width;
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000: {
            bytes = ((cudaMallocManaged_v6000_params_st*)(params))->size;
            break;
        }
        default: {
            return false;
        }
    }
    double value = (double)(bytes);
    store_counter_data("GPU: Bytes Allocated", context, apex::profiler::get_time_ns(), value);
    return true;
}

void apex_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain,
        CUpti_CallbackId id, const void *params) {
    static bool initialized = initialize_first_time();
    APEX_UNUSED(initialized);
    /* Supposedly, we can use the ud or cbdata->contextData fields
     * to pass data from the start to the end event, but it isn't
     * broadly supported in the CUPTI interface, so we'll manage the
     * timer stack locally. */
    static APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > timer_stack;
    APEX_UNUSED(ud);
    APEX_UNUSED(id);
    APEX_UNUSED(domain);
    if (!apex::thread_instance::is_worker()) { return; }
    if (params == NULL) { return; }
    CUpti_CallbackData * cbdata = (CUpti_CallbackData*)(params);

    if (cbdata->callbackSite == CUPTI_API_ENTER) {
        std::stringstream ss;
        ss << cbdata->functionName;
        if (apex::apex_options::use_cuda_kernel_details()) {
            if (cbdata->symbolName != NULL && strlen(cbdata->symbolName) > 0) {
                ss << ": " << cbdata->symbolName;
            }
        }
        std::string tmp(ss.str());
        /*
           std::string tmp(cbdata->functionName);
           */
        auto timer = apex::new_task(tmp);
        apex::start(timer);
        timer_stack.push(timer);
        map_mutex.lock();
        correlation_map[cbdata->correlationId] = timer;
        map_mutex.unlock();
        getBytesIfMalloc(id, cbdata->functionParams, tmp);
    } else if (!timer_stack.empty()) {
        auto timer = timer_stack.top();
        apex::stop(timer);
        timer_stack.pop();
    }
}

void initTrace() {
    bool& registered = get_registered();
    registered = true;

    size_t attrValue = 0, attrValueSize = sizeof(size_t);
    // Device activity record is created when CUDA initializes, so we
    // want to enable it before cuInit() or any CUDA runtime call.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE)); // 8
    // Enable all other activity record kinds.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)); // 10
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)); // 1
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2)); // 22
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET)); // 2
#if 0
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)); // 3   <- disables concurrency
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER)); // 4
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)); // 5
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EVENT)); // 6
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_METRIC)); // 7
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
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA)); // 33
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH)); // 34
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER)); // 35
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
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENMP)); // 47
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API)); // 48
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_COUNT)); // 49
#endif
#if 0 // not until CUDA 11
#endif

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity API.
    // Some attributes require to be set before any CUDA context is created to be effective,
    // e.g. to be applied to all device buffer allocations (see documentation).
    CUPTI_CALL(cuptiActivityGetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
    attrValue = attrValue / 4;
    /*
    printf("%s = %llu\n",
            "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
            (long long unsigned)attrValue);
            */
    CUPTI_CALL(cuptiActivitySetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiActivityGetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
    attrValue = attrValue * 4;
    /*
    printf("%s = %llu\n",
            "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
            (long long unsigned)attrValue);
            */
    CUPTI_CALL(cuptiActivitySetAttribute(
                CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

    /* now that the activity is configured, subscribe to callback support, too. */
    CUPTI_CALL(cuptiSubscribe(&subscriber,
                (CUpti_CallbackFunc)apex_cupti_callback_dispatch, NULL));
    // get device callbacks
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    /* These events aren't begin/end callbacks, so no need to support them. */
    //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE));
    //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE));
    //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX));

    // synchronize timestamps
    startTimestampCPU = apex::profiler::get_time_ns();
    cuptiGetTimestamp(&startTimestampGPU);
    // assume CPU timestamp is greater than GPU
    deltaTimestamp = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    printf("Delta computed to be: %ld\n", deltaTimestamp);
}

/* This is the global "shutdown" method for flushing the buffer.  This is
 * called from apex::finalize().  It's the only function in the CUDA support
 * that APEX will call directly. */
namespace apex {
    void flushTrace(void) {
        if ((num_buffers_processed + 10) < num_buffers) {
            flushing = true;
            std::cout << "Flushing remaining " << std::fixed
                << num_buffers-num_buffers_processed << " of " << num_buffers
                << " CUDA/CUPTI buffers..." << std::endl;
        }
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
        if (flushing) {
            std::cout << std::endl;
        }
    }
}
