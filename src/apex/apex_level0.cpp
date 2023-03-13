//==============================================================
// Copyright © 2020 Intel Corporation, 2022 University of Oregon
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <cstring>
#include <fstream>
#include <regex>
#include <unistd.h>
#include <L0/utils.h>
#include <L0/ze_kernel_collector.h>
#include <L0/ze_api_collector.h>
#include "apex_api.hpp"
#include "apex.hpp"
#include "trace_event_listener.hpp"
#if defined(APEX_WITH_PERFETTO)
#include "perfetto_listener.hpp"
#endif

using namespace std;
using namespace apex;

static ZeApiCollector* api_collector = nullptr;
static ZeKernelCollector* kernel_collector = nullptr;
static chrono::steady_clock::time_point start_time;
static uint64_t cpu_delta = 0L;
static uint64_t gpu_delta = 0L;
static uint64_t last_gpu_timestamp = 0L;
static uint64_t gpu_offset = 0L;
static uint64_t device_resolution{0L};
static uint64_t device_mask{0L};

// External Tool Interface ////////////////////////////////////////////////////

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
void Usage() {
  cout <<
    "Usage: ./ze_hot_kernels[.exe] <application> <args>" <<
    endl;
}

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
int ParseArgs(int argc, char* argv[]) {
  return 1;
}

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
void SetToolEnv() {
  utils::SetEnv("ZET_ENABLE_API_TRACING_EXP","1");
}

// Internal Tool Functionality ////////////////////////////////////////////////

namespace apex {
namespace level0 {

/*
taken from: https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
Time Correlation

Common problem while kernel timestamps collection is to map these timestamps
to general CPU timeline. Since Level Zero provides kernel timestamps in GPU
clocks, one may need to convert them to some CPU time. Starting from Level
Zero 1.1, new function zeDeviceGetGlobalTimestamps is available. Using this
function, one can get correlated host (CPU) and device (GPU) timestamps for
any particular device:

    uint64_t host_timestamp = 0, device_timestamp = 0;
    ze_result_t status = zeDeviceGetGlobalTimestamps(
        device, &host_timestamp, &device_timestamp);
    assert(status == ZE_RESULT_SUCCESS);

Host timestamp value corresponds to CLOCK_MONOTONIC_RAW on Linux or
QueryPerformanceCounter on Windows, while device timestamp for GPU is
collected in raw GPU cycles.

Note that the number of valid bits for the device timestamp returned by
zeDeviceGetGlobalTimestamps is timestampValidBits, while the global kernel
timastamp returned by zeEventQueryKernelTimestamp has kernelTimestampValidBits
(both values are fields of ze_device_properties_t). And currently
kernelTimestampValidBits is less then timestampValidBits, so to map kernels
into CPU timeline one may need to truncate device timestamp to
kernelTimestampValidBits:

    ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, };
    ze_result_t status = zeDeviceGetProperties(device, &props);
    assert(status == ZE_RESULT_SUCCESS);
    uint64_t mask = (1ull << props.kernelTimestampValidBits) - 1ull;
    uint64_t kernel_timestamp = (device_timestamp & mask);

To convert GPU cycles into seconds one may use timerResolution field from
ze_device_properties_t structure, that represents cycles per second starting
from Level Zero 1.2:

    ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, };
    ze_result_t status = zeDeviceGetProperties(device, &props);
    assert(status == ZE_RESULT_SUCCESS);
    const uint64_t NSEC_IN_SEC = 1000000000;
    uint64_t device_timestamp_ns = NSEC_IN_SEC * device_timestamp / props.timerResolution;

*/

/* That said, all the timestamps are converted for us in ze_kernel_collector.h.
   However, we need to apply a delta.
   */

void OnAPIFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {
  uint64_t taskid;

  taskid = *((uint64_t *) data);
  /*
  DEBUG_PRINT("APEX: OnAPIFinishCallback:        (raw) name: %s started: %lu ended: %lu at: %lu task id=%lu\n",
		  name.c_str(), started, ended, profiler::now_ns(), taskid);
          */

    // create a task_wrapper, as a child of the current timer
    auto tt = new_task(name, UINT64_MAX, nullptr);
    // create an APEX profiler to store this data - we can't start
    // then stop because we have timestamps already.
    auto prof = std::make_shared<profiler>(tt);
    prof->set_start(started);
    prof->set_end(ended);
    // important!  Otherwise we might get the wrong end timestamp.
    prof->stopped = true;
    // Get the singleton APEX instance
    static auto * instance = ::apex::apex::instance();
    // fake out the profiler_listener
    instance->the_profiler_listener->push_profiler_public(prof);
    // Handle tracing, if necessary
    if (apex_options::use_trace_event()) {
        trace_event_listener * tel =
            (trace_event_listener*)instance->the_trace_event_listener;
        tel->on_stop(prof);
    }
#if defined(APEX_WITH_PERFETTO)
    if (apex_options::use_perfetto()) {
        perfetto_listener * tel =
            (perfetto_listener*)instance->the_perfetto_listener;
        tel->on_start(tt);
        tel->on_stop(prof);
    }
#endif
#ifdef APEX_HAVE_OTF2
    if (apex_options::use_otf2()) {
        otf2_listener * tol =
            (otf2_listener*)instance->the_otf2_listener;
        tol->on_start(prof);
        tol->on_stop(prof);
    }
#endif
    // have the listeners handle the end of this task
    instance->complete_task(tt);
}

void store_profiler_data(const std::string &name,
        uint64_t start, uint64_t end, level0_thread_node &node,
        std::shared_ptr<task_wrapper> parent) {
    in_apex prevent_deadlocks;
    async_event_data as_data;
    as_data.flow = false;
    // create a task_wrapper, as a GPU child of the parent on the CPU side
    auto tt = new_task(name, UINT64_MAX, parent);
    // create an APEX profiler to store this data - we can't start
    // then stop because we have timestamps already.
    auto prof = std::make_shared<profiler>(tt);
    prof->set_start(start);
    prof->set_end(end);
    // important!  Otherwise we might get the wrong end timestamp.
    prof->stopped = true;
    // Get the singleton APEX instance
    static auto* instance = ::apex::apex::instance();
    // fake out the profiler_listener
    instance->the_profiler_listener->push_profiler_public(prof);
    // Handle tracing, if necessary
    if (apex_options::use_trace_event()) {
        trace_event_listener * tel =
            (trace_event_listener*)instance->the_trace_event_listener;
        tel->on_async_event(node, prof, as_data);
    }
#if defined(APEX_WITH_PERFETTO)
    if (apex_options::use_perfetto()) {
        perfetto_listener * tel =
            (perfetto_listener*)instance->the_perfetto_listener;
        tel->on_async_event(node, prof, as_data);
    }
#endif
#ifdef APEX_HAVE_OTF2
    if (apex_options::use_otf2()) {
        otf2_listener * tol =
            (otf2_listener*)instance->the_otf2_listener;
        tol->on_async_event(node, prof);
    }
#endif
    // have the listeners handle the end of this task
    instance->complete_task(tt);
}


void OnKernelFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {

  int taskid;
  taskid = *((int *) data);
  /* We get a start and stop timestamp from the API in nanoseconds - but they
     only make sense relative to each other. however, we're getting a callback
     at exactly the time the kernel finishes, so we can assume the end time is
     now, and then take a delta from now for the start time. */
  uint64_t ended_translated = profiler::now_ns();
  uint64_t started_translated = ended_translated - (ended - started);
  /*
  DEBUG_PRINT("APEX: <kernel>: (raw) name: %s started: %20lu ended: %20lu at: %20lu task id=%d\n",
		  name.substr(0,10).c_str(), started, ended, profiler::now_ns(), taskid);
  DEBUG_PRINT("APEX: <kernel>: (raw) name: %s started: %20lu ended: %20lu at: %20lu task id=%d\n",
    name.substr(0,10).c_str(),  started_translated, ended_translated, profiler::now_ns(), taskid);
    */

  last_gpu_timestamp = ended;
  int device_num = 0;
  int parent_thread = 0;
  std::string demangled = demangle(name);
  demangled = regex_replace(demangled, regex("typeinfo name for "), "GPU: ");
  level0_thread_node node(device_num, parent_thread, APEX_ASYNC_KERNEL);
  store_profiler_data(demangled, started_translated, ended_translated, node, nullptr);

  return;
}


// Internal Tool Interface ////////////////////////////////////////////////////

void EnableProfiling() {
  if (getenv("ZE_ENABLE_TRACING_LAYER") == NULL) {
    // tau_exec -level_zero was not called. Perhaps it is using -opencl
    DEBUG_PRINT("APEX: Disabling Level Zero support as ZE_ENABLE_TRACING_LAYER was not set from tau_exec -l0\n");
    return;
  }
  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  driver =  utils::ze::GetGpuDriver();
  device =  utils::ze::GetGpuDevice();

  if (device == nullptr || driver == nullptr) {
    std::cout << "[WARNING] Unable to find target device" << std::endl;
    return;
  }

    // register a callback for Kernel calls
  uint64_t *kernel_taskid = new uint64_t;
  void *pk = (void *) kernel_taskid;
  kernel_collector = ZeKernelCollector::Create(driver,
                  OnKernelFinishCallback, pk);

  // For API calls, we create a new task and trigger the start/stop based on its
  // timestamps.

  uint64_t *api_taskid  = new uint64_t;
  void *ph = (void *) api_taskid;
  api_collector = ZeApiCollector::Create(driver, OnAPIFinishCallback, ph);

  start_time = std::chrono::steady_clock::now();
}

void DisableProfiling() {
  static bool once{false};
  if (once) return;
  once = true;
  if (kernel_collector != nullptr) {
    kernel_collector->DisableTracing();
    delete kernel_collector;
  }
  if (api_collector != nullptr) {
    api_collector->DisableTracing();
    delete api_collector;
  }
}

} // namespace level0
} // namespace apex


