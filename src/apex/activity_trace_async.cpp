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
#include "apex.hpp"

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
static uint64_t startTimestamp;

/* The callback subscriber */
CUpti_SubscriberHandle subscriber;

/* The map that holds correlation IDs and matches them to GUIDs */
std::unordered_map<uint32_t, uint64_t> correlation_map;
std::mutex map_mutex;

void store_profiler_data(const char * name, uint64_t start, uint64_t end) {
      static apex::apex* instance = apex::apex::instance();
      // get the elapsed time and scale to nanoseconds
      double nanoseconds = ((double)(end - start)/1000.0);
      // create an APEX profiler to store this data
      std::shared_ptr<apex::profiler> p =
        std::make_shared<apex::profiler>(apex::task_identifier::get_task_id(
        name), nanoseconds, true);
      instance->the_profiler_listener->push_profiler_public(p);
}

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "GPU: Memory Copy HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "GPU: Memory Copy DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "GPU: Memory Copy HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "GPU: Memory Copy AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "GPU: Memory Copy AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "GPU: Memory Copy AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "GPU: Memory Copy DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "GPU: Memory Copy DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "GPU: Memory Copy HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "GPU: COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "GPU: BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "GPU: INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "GPU: RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "GPU: PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "GPU: THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "GPU: DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "GPU: CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "GPU: STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

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

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
      printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
      printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
             context->contextId, context->deviceId,
             getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
             (int) context->nullStreamId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
  case CUPTI_ACTIVITY_KIND_MEMCPY2:
    {
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
#if 0
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
             (unsigned long long) (memcpy->start - startTimestamp),
             (unsigned long long) (memcpy->end - startTimestamp),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
#endif
      store_profiler_data(getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind), memcpy->start, memcpy->end);
      break;
    }
#if 0 // not until CUDA 11
  case CUPTI_ACTIVITY_KIND_MEMCPY2:
    {
      CUpti_ActivityMemcpyPtoP *memcpy = (CUpti_ActivityMemcpyPtoP *) record;
      printf("MEMCPY2 %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
             (unsigned long long) (memcpy->start - startTimestamp),
             (unsigned long long) (memcpy->end - startTimestamp),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
      store_profiler_data(getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind), memcpy->start, memcpy->end);
      break;
    }
#endif
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
#if 0
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             memset->value,
             (unsigned long long) (memset->start - startTimestamp),
             (unsigned long long) (memset->end - startTimestamp),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId);
#endif
      store_profiler_data("GPU: Memset", memset->start, memset->end);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
  case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
    {
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record;
#if 0
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      printf("%f\n", nanoseconds);
#endif
      std::stringstream ss;
      ss << "GPU: " << kernel->name;
      store_profiler_data(ss.str().c_str(), kernel->start, kernel->end);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        printf("NAME  %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      case CUPTI_ACTIVITY_OBJECT_STREAM:
        printf("NAME %s %u %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      default:
        printf("NAME %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               name->name);
        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER id %u [ %llu ], name %s, domain %s\n",
             marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA:
    {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
             marker->id, marker->color, marker->category,
             (unsigned long long) marker->payload.metricValueUint64,
             marker->payload.metricValueDouble);
      break;
    }
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
      printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
             getActivityOverheadKindString(overhead->overheadKind),
             (unsigned long long) overhead->start - startTimestamp,
             (unsigned long long) overhead->end - startTimestamp,
             getActivityObjectKindString(overhead->objectKind),
             getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
      break;
    }
  default:
    printf("  <unknown>\n");
    break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
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

void apex_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain,
    CUpti_CallbackId id, const void *params) {
    /* Supposedly, we can use the ud or cbdata->contextData fields
     * to pass data from the start to the end event, but it isn't
     * broadly supported in the CUPTI interface, so we'll manage the
     * timer stack locally. */
    thread_local static std::stack<apex::profiler*> timer_stack;
    APEX_UNUSED(ud);
    APEX_UNUSED(id);
    APEX_UNUSED(domain);
    if (params == NULL) return;
    apex::profiler *p = nullptr;
    CUpti_CallbackData * cbdata = (CUpti_CallbackData*)(params);

    if (cbdata->callbackSite == CUPTI_API_ENTER) {
        // check for a memory copy event
        if (cbdata->symbolName != NULL) {
            std::stringstream ss;
            ss << cbdata->functionName << ": " << cbdata->symbolName;
            p = apex::start(ss.str());
        } else {
            p = apex::start(cbdata->functionName);
        }
        timer_stack.push(p);
    } else if (!timer_stack.empty()) {
        p = timer_stack.top();
        apex::stop(p);
        timer_stack.pop();
    }
}

void
initTrace()
{
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
  printf("%s = %llu\n",
    "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
    (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiActivityGetAttribute(
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n",
    "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
        (long long unsigned)attrValue);
  attrValue *= 2;
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

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

/* This is the global "shutdown" method for flushing the buffer.  This is
 * called from apex::finalize().  It's the only function in the CUDA support
 * that APEX will call directly. */
namespace apex {
    void flushTrace(void) {
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
    }
}
