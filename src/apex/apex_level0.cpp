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
#include <L0/utils.h>
#include <L0/ze_kernel_collector.h>
#include <L0/ze_api_collector.h>
#include "apex_api.hpp"
#include "apex.hpp"
#include "trace_event_listener.hpp"

using namespace std;
using namespace apex;

static ZeApiCollector* api_collector = nullptr;
static ZeKernelCollector* kernel_collector = nullptr;
static chrono::steady_clock::time_point start_time;
static int gpu_task_id = 0;
static int host_api_task_id = 0;
static uint64_t first_clock_timestamp;
static uint64_t first_cpu_timestamp;
static uint64_t first_gpu_timestamp;
static uint64_t cpu_delta = 0L;
static uint64_t gpu_delta = 0L;
static uint64_t last_gpu_timestamp = 0L;
static uint64_t gpu_offset = 0L;

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

static void PrintResults() {
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  chrono::duration<uint64_t, nano> time = end - start_time;

  PTI_ASSERT(kernel_collector != nullptr);
  const ZeKernelInfoMap& kernel_info_map = kernel_collector->GetKernelInfoMap();
  if (kernel_info_map.size() == 0) {
    return;
  }

  uint64_t total_duration = 0;
  for (auto& value : kernel_info_map) {
    total_duration += value.second.total_time;
  }

  cerr << endl;
  cerr << "=== Device Timing Results: ===" << endl;
  cerr << endl;
  cerr << "Total Execution Time (ns): " << time.count() << endl;
  cerr << "Total Device Time (ns): " << total_duration << endl;
  cerr << endl;

  if (total_duration > 0) {
    ZeKernelCollector::PrintKernelsTable(kernel_info_map);
  }

  cerr << endl;
}

// Internal Tool Functionality ////////////////////////////////////////////////

static void APIPrintResults() {
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  chrono::duration<uint64_t, nano> time = end - start_time;

  PTI_ASSERT(api_collector != nullptr);
  const ZeFunctionInfoMap& function_info_map = api_collector->GetFunctionInfoMap();
  if (function_info_map.size() == 0) {
    return;
  }

  uint64_t total_duration = 0;
  for (auto& value : function_info_map) {
    total_duration += value.second.total_time;
  }

  cerr << endl;
  cerr << "=== API Timing Results: ===" << endl;
  cerr << endl;
  cerr << "Total Execution Time (ns): " << time.count() << endl;
  cerr << "Total API Time (ns): " << total_duration << endl;
  cerr << endl;

  if (total_duration > 0) {
    ZeApiCollector::PrintFunctionsTable(function_info_map);
  }

  std::cerr << std::endl;
}

uint64_t TAUTranslateGPUTimestamp(uint64_t gpu_ts) {
  // gpu_ts is in nanoseconds.
  uint64_t new_ts = gpu_ts + gpu_delta;
  return new_ts;
}

uint64_t TAUTranslateCPUTimestamp(uint64_t cpu_ts) {
  // cpu_ts is in nanoseconds.
  uint64_t new_ts = cpu_ts + cpu_delta;
  return new_ts;
}

void TAUOnAPIFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {
  uint64_t taskid;

  taskid = *((uint64_t *) data);
  uint64_t started_translated = TAUTranslateCPUTimestamp(started);
  uint64_t ended_translated = TAUTranslateCPUTimestamp(ended);
  DEBUG_PRINT("APEX: OnAPIFinishCallback: (raw) name: %s started: %lu ended: %lu task id=%lu\n",
		  name.c_str(), started, ended, taskid);
  DEBUG_PRINT("APEX: OnAPIFinishCallback: (translated) name: %s started: %lu ended: %lu task id=%lu\n",
		  name.c_str(), started_translated, ended_translated, taskid);
  // We now need to start a timer on a task at the started_translated time and end at ended_translated

    // create a task_wrapper, as a child of the current timer
    auto tt = new_task(name, UINT64_MAX, nullptr);
    // create an APEX profiler to store this data - we can't start
    // then stop because we have timestamps already.
    auto prof = std::make_shared<profiler>(tt);
    prof->set_start(started_translated);
    prof->set_end(ended_translated);
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


void TAUOnKernelFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {

  int taskid;
  taskid = *((int *) data);
  uint64_t started_translated = TAUTranslateGPUTimestamp(started);
  uint64_t ended_translated = TAUTranslateGPUTimestamp(ended);
  DEBUG_PRINT("APEX: <kernel>: (raw) name: %s  started: %lu ended: %lu task id=%d\n",
		  name.c_str(), started, ended, taskid);
  DEBUG_PRINT("APEX: <kernel>: (raw) name: %s started: %lu ended: %lu task id=%d\n",
    name.c_str(),  started_translated, ended_translated, taskid);

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

  uint64_t *kernel_taskid = new uint64_t;
  //TAU_CREATE_TASK(*kernel_taskid);
  void *pk = (void *) kernel_taskid;
  gpu_task_id = *kernel_taskid;
  uint64_t *api_taskid  = new uint64_t;
  //*host_taskid = RtsLayer::myThread();
  //TAU_CREATE_TASK(*api_taskid);
  host_api_task_id = *api_taskid;
  kernel_collector = ZeKernelCollector::Create(driver,
                  TAUOnKernelFinishCallback, pk);
  /*
  //uint64_t gpu_ts = utils::i915::GetGpuTimestamp() & 0x0FFFFFFFF;
  uint64_t gpu_ts = utils::i915::GetGpuTimestamp() ;
  std::cout <<"TAU: Earliest GPU timestamp "<<gpu_ts<<std::endl;
  */
  first_cpu_timestamp = 0L;
  first_gpu_timestamp = 0L;
  utils::ze::GetDeviceTimestamps(device, &first_cpu_timestamp, &first_gpu_timestamp);
  first_clock_timestamp = profiler::now_ns();
  cpu_delta = first_clock_timestamp - first_cpu_timestamp;
  gpu_delta = first_clock_timestamp - first_gpu_timestamp;
  DEBUG_PRINT("APEX: First CPU timestamp= %ld \n",first_cpu_timestamp);
  DEBUG_PRINT("APEX: First GPU timestamp= %ld \n",first_gpu_timestamp);
  DEBUG_PRINT("APEX: Real CPU timestamp= %ld \n",first_clock_timestamp);
  DEBUG_PRINT("APEX: CPU delta= %ld \n",cpu_delta);
  DEBUG_PRINT("APEX: GPU delta= %ld \n",gpu_delta);
  utils::ze::GetDeviceTimestamps(device, &first_cpu_timestamp, &first_gpu_timestamp);
  first_clock_timestamp = profiler::now_ns();
  DEBUG_PRINT("APEX: Second CPU timestamp= %ld \n",first_cpu_timestamp);
  DEBUG_PRINT("APEX: Second GPU timestamp= %ld \n",first_gpu_timestamp);
  DEBUG_PRINT("APEX: Real CPU timestamp= %ld \n",first_clock_timestamp);
  utils::ze::GetDeviceTimestamps(device, &first_cpu_timestamp, &first_gpu_timestamp);
  first_clock_timestamp = profiler::now_ns();
  DEBUG_PRINT("APEX: Third CPU timestamp= %ld \n",first_cpu_timestamp);
  DEBUG_PRINT("APEX: Third GPU timestamp= %ld \n",first_gpu_timestamp);
  DEBUG_PRINT("APEX: Real CPU timestamp= %ld \n",first_clock_timestamp);
  utils::ze::GetDeviceTimestamps(device, &first_cpu_timestamp, &first_gpu_timestamp);
  first_clock_timestamp = profiler::now_ns();
  DEBUG_PRINT("APEX: Fourth CPU timestamp= %ld \n",first_cpu_timestamp);
  DEBUG_PRINT("APEX: Fourth GPU timestamp= %ld \n",first_gpu_timestamp);
  DEBUG_PRINT("APEX: Real CPU timestamp= %ld \n",first_clock_timestamp);
  utils::ze::GetDeviceTimestamps(device, &first_cpu_timestamp, &first_gpu_timestamp);
  first_clock_timestamp = profiler::now_ns();
  DEBUG_PRINT("APEX: Fifth CPU timestamp= %ld \n",first_cpu_timestamp);
  DEBUG_PRINT("APEX: Fifth GPU timestamp= %ld \n",first_gpu_timestamp);
  DEBUG_PRINT("APEX: Real CPU timestamp= %ld \n",first_clock_timestamp);

  // For API calls, we create a new task and trigger the start/stop based on its
  // timestamps.

  void *ph = (void *) api_taskid;
  api_collector = ZeApiCollector::Create(driver, TAUOnAPIFinishCallback, ph);

  start_time = std::chrono::steady_clock::now();
}

void DisableProfiling() {
  if (kernel_collector != nullptr) {
    kernel_collector->DisableTracing();
    //if (TauEnv_get_verbose())
      PrintResults();
    delete kernel_collector;
  }
  if (api_collector != nullptr) {
    api_collector->DisableTracing();
    //if (TauEnv_get_verbose())
      APIPrintResults();
    delete api_collector;
  }
  //uint64_t gpu_end_ts = utils::i915::GetGpuTimestamp() & 0x0FFFFFFFF;
  /*
  uint64_t gpu_end_ts = utils::i915::GetGpuTimestamp();
  std::cout <<"APEX: Latest GPU timestamp "<<gpu_end_ts<<std::endl;
  */
  int taskid = gpu_task_id;  // GPU task id is 1;
  uint64_t last_gpu_translated = TAUTranslateGPUTimestamp(last_gpu_timestamp);
  DEBUG_PRINT("APEX: Latest GPU timestamp (raw) =%ld\n", last_gpu_timestamp);
  DEBUG_PRINT("APEX: Latest GPU timestamp (translated) =%ld\n",last_gpu_translated);
  uint64_t cpu_end_ts = profiler::now_ns();
  // metric_set_gpu_timestamp(taskid, last_gpu_translated);
  //Tau_stop_top_level_timer_if_necessary_task(taskid);

  // metric_set_gpu_timestamp(host_api_task_id, cpu_end_ts);
  //Tau_create_top_level_timer_if_necessary_task(host_api_task_id);

  DEBUG_PRINT("APEX: Latest CPU timestamp =%ld\n", cpu_end_ts);
  std::chrono::steady_clock::time_point chrono_end = std::chrono::steady_clock::now();
  std::chrono::duration<uint64_t, std::nano> chrono_dt = chrono_end - start_time;
  DEBUG_PRINT("APEX: Diff (chrono) =%ld \n", chrono_dt.count());
}


// preload.cc
#if defined(__gnu_linux__)

#include <dlfcn.h>

typedef void (*Exit)(int status) __attribute__ ((noreturn));
typedef int (*Main)(int argc, char** argv, char** envp);
typedef int (*Fini)(void);
typedef int (*LibcStartMain)(Main main, int argc, char** argv, Main init,
                             Fini fini, Fini rtld_fini, void *stack_end);

// Pointer to original application main() function
Main original_main = nullptr;

extern "C" int HookedMain(int argc, char **argv, char **envp) {
  EnableProfiling();
  int return_code = original_main(argc, argv, envp);
  DisableProfiling();
  return return_code;
}

extern "C" int __libc_start_main(Main main,
                                 int argc,
                                 char** argv,
                                 Main init,
                                 Fini fini,
                                 Fini rtld_fini,
                                 void* stack_end) {
  original_main = main;
  LibcStartMain original =
    (LibcStartMain)dlsym(RTLD_NEXT, "__libc_start_main");
  return original(HookedMain, argc, argv, init, fini, rtld_fini, stack_end);
}

extern "C" void exit(int status) {
  Exit original = (Exit)dlsym(RTLD_NEXT, "exit");
  DisableProfiling();
  original(status);
}

#else
#error not supported
#endif

