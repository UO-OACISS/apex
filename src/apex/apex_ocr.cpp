#ifdef APEX_HAVE_OCR

#include <map>
#include <iostream>
#include <cinttypes>
#include <cstdio>
#include <atomic>
#include <utility>
#include <stack>
#include <sstream>
#include "apex_api.hpp"
#include "apex_types.h"
#include "thread_instance.hpp"

#define OCR_TYPE_H x86.h
#include "ocr.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#ifdef APEX_OCR_DEBUG
#define DEBUG_MSG(str) std::cerr << str << std::endl
#else
#define DEBUG_MSG(str)
#endif

inline bool operator<(const ocrGuid_t & left, const ocrGuid_t & right) {
    return ocrGuidIsLt(left, right);
}

static const std::string guid_to_str(ocrGuid_t guid) {
    constexpr size_t max = 256;
    char name_arr[max];
    std::snprintf(name_arr, max, GUIDF, GUIDA(guid));
    std::string name{name_arr};
    return name;
}

using guid_map_t = std::map<ocrGuid_t, apex::profiler*>;

static APEX_NATIVE_TLS guid_map_t * guid_map = nullptr;
static APEX_NATIVE_TLS bool thread_seen = false;
static std::atomic_bool pd_seen{false};
static std::atomic<uint64_t> thread_id{0};

static inline guid_map_t * get_guid_map() {
    if(unlikely(guid_map == nullptr)) {
        guid_map = new guid_map_t();
    }
    return guid_map;
}

static inline void check_registration(const int node) {
    if(unlikely(!thread_seen)) {
        if(unlikely(!pd_seen)) {
            pd_seen = true;
            apex::set_node_id(node);
        }
        std::stringstream ss;
        ss << "worker " << thread_id++;
        apex::register_thread(ss.str());
        thread_seen = true;
    }
}

static inline void apex_ocr_init() {
    DEBUG_MSG("APEX OCR init");
    apex::init("main");
}

static inline void apex_ocr_exit_thread() {
    DEBUG_MSG("APEX OCR exit thread");
    apex::exit_thread();
    if(guid_map != nullptr) {
        delete guid_map;
        guid_map = nullptr;
    }
    thread_seen = false;
}

static inline void apex_ocr_shutdown() {
    DEBUG_MSG("APEX OCR shutdown");
    apex_ocr_exit_thread();
    apex::finalize();
}

static inline void apex_ocr_task_create(const ocrGuid_t edtGuid, const apex_function_address fctPtr, const int node) {
    DEBUG_MSG("Task created: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::new_task(task_id, nullptr);    
} 

static inline void apex_ocr_task_destroy(const ocrGuid_t edtGuid, const apex_function_address fctPtr, const int node) {
    DEBUG_MSG("Task destroyed: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::destroy_task(task_id, nullptr);
}


static inline void apex_ocr_task_runnable(const ocrGuid_t edtGuid, const apex_function_address fctPtr, const int node) {
    DEBUG_MSG("Task runnable: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::set_task_state(task_id, APEX_TASK_ELIGIBLE);
}

static inline void apex_ocr_task_add_dependence(const ocrGuid_t src, const ocrGuid_t dest, const int node) {
    DEBUG_MSG("Task add dependence: " << guid_to_str(src) << " --> " << guid_to_str(dest));
    check_registration(node);
    apex::task_identifier * src_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(src));
    apex::task_identifier * dest_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dest));
    apex::new_dependency(src_id, dest_id);

}


static inline void apex_ocr_task_satisfy_dependence(const ocrGuid_t edtGuid, const ocrGuid_t satisfyee, const int node) {
    DEBUG_MSG("Task satisfy dependence. Task: " << guid_to_str(edtGuid) << " Satisfyee: " << guid_to_str(satisfyee));
    check_registration(node);
    apex::task_identifier * task_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(edtGuid));
    apex::task_identifier * satisfyee_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(satisfyee));
    apex::satisfy_dependency(task_id, satisfyee_id);
}

static inline void apex_ocr_task_execute(const ocrGuid_t edtGuid, const apex_function_address fctPtr, const int node) {
    DEBUG_MSG("Task executing: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::profiler * p = apex::start(task_id);
    (*get_guid_map())[edtGuid] = p;
}

static inline void apex_ocr_task_finish(const ocrGuid_t edtGuid, const int node) {
    DEBUG_MSG("Task finished: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::profiler * p = (*get_guid_map())[edtGuid];
#ifdef APEX_OCR_DEBUG
    if(p == nullptr) {
        std::cerr << "apex_ocr_task_finish: Profiler null for " << guid_to_str(edtGuid) << std::endl;
        abort();
    }
#endif
    apex::stop(p);
    get_guid_map()->erase(edtGuid);
}

static inline void apex_ocr_task_yield(const ocrGuid_t edtGuid, const apex_function_address fctPtr, const int node) { 
    DEBUG_MSG("Task yielded: " << guid_to_str(edtGuid));
    check_registration(node);
    apex::profiler * p = (*get_guid_map())[edtGuid];
#ifdef APEX_OCR_DEBUG
    if(p == nullptr) {
        std::cerr << "apex_ocr_task_yield: Profiler null for " << guid_to_str(edtGuid) << std::endl;
        abort();
    }
#endif
    apex::yield(p);
    get_guid_map()->erase(edtGuid);
}

static inline void apex_ocr_task_data_acquire(const ocrGuid_t edtGuid, const ocrGuid_t dbGuid, const u64 dbSize, const int node) {
    DEBUG_MSG("Task data acquire. Task " << guid_to_str(edtGuid) << " acquired DB " << guid_to_str(dbGuid) << " of size " << dbSize);
    check_registration(node);
    apex::task_identifier * task_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(edtGuid));
    apex::task_identifier * data_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dbGuid), APEX_DATA_ID);
    apex::acquire_data(task_id, data_id, dbSize);
}

static inline void apex_ocr_task_data_release(const ocrGuid_t edtGuid, const ocrGuid_t dbGuid, const u64 dbSize, const int node) {
    DEBUG_MSG("Task data release. Task " << guid_to_str(edtGuid) << " released DB " << guid_to_str(dbGuid) << " of size " << dbSize);
    check_registration(node);
    apex::task_identifier * task_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(edtGuid));
    apex::task_identifier * data_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dbGuid), APEX_DATA_ID);
    DEBUG_MSG("task_id: " << task_id << ", data_id: " << data_id);
    apex::release_data(task_id, data_id, dbSize);
}

static inline void apex_ocr_event_create(const ocrGuid_t eventGuid, const int node) {
    DEBUG_MSG("Event create: " << guid_to_str(eventGuid));
    check_registration(node);
    apex::task_identifier * event_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(eventGuid), APEX_EVENT_ID);
    apex::new_event(event_id);                                                                            
}

static inline void apex_ocr_event_destroy(const ocrGuid_t eventGuid, const int node) {
    DEBUG_MSG("Event destroy: " << guid_to_str(eventGuid));
    check_registration(node);
    apex::task_identifier * event_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(eventGuid), APEX_EVENT_ID);
    apex::destroy_event(event_id);                                                                            
}

static inline void apex_ocr_event_satisfy_dependence(const ocrGuid_t eventGuid, const ocrGuid_t satisfyee, const int node) {
    DEBUG_MSG("Event satisfy dependence. Event: " << guid_to_str(eventGuid) << " Satisfyee: " << guid_to_str(satisfyee));
    check_registration(node);
    apex::task_identifier * event_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(eventGuid), APEX_EVENT_ID);
    apex::task_identifier * satisfyee_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(satisfyee));
    apex::new_dependency(event_id, satisfyee_id);
}

static inline void apex_ocr_event_add_dependence(const ocrGuid_t src, const ocrGuid_t dest, const int node) {
    DEBUG_MSG("Event add dependence: " << guid_to_str(src) << " --> " << guid_to_str(dest));
    check_registration(node);
    apex::task_identifier * src_id  = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(src), APEX_EVENT_ID);
    apex::task_identifier * dest_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dest));
    apex::new_dependency(src_id, dest_id);
}

static inline void apex_ocr_data_create(const ocrGuid_t dbGuid, const u64 dbSize, const int node) {
    DEBUG_MSG("Data create: " << guid_to_str(dbGuid) << " of size " << dbSize);
    check_registration(node);
    apex::task_identifier * data_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dbGuid), APEX_DATA_ID);
    apex::new_data(data_id, dbSize);
}

static inline void apex_ocr_data_destroy(const ocrGuid_t dbGuid, const int node) {
    DEBUG_MSG("Data destroy: " << guid_to_str(dbGuid));
    check_registration(node);
    apex::task_identifier * data_id = new apex::task_identifier((apex_function_address)APEX_NULL_FUNCTION_ADDRESS, guid_to_str(dbGuid), APEX_DATA_ID);
    apex::destroy_data(data_id);
}

extern "C" {

void traceTaskCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                     ocrEdt_t fctPtr) { 
    apex_ocr_task_create(edtGuid, (apex_function_address)fctPtr, (int)location);
}



void traceTaskDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                      ocrEdt_t fctPtr) {
    apex_ocr_task_destroy(edtGuid, (apex_function_address)fctPtr, (int)location);
}



void traceTaskRunnable(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                       ocrEdt_t fctPtr) { 
    apex_ocr_task_runnable(edtGuid, (apex_function_address)fctPtr, (int)location);
}



void traceTaskAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                            ocrTraceAction_t actionType, u64 workerId,
                            u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                            ocrGuid_t dest) {                                                                    
    apex_ocr_task_add_dependence(src, dest, (int)location);
}



void traceTaskSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                ocrTraceAction_t actionType, u64 workerId,
                                u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                                ocrGuid_t satisfyee) { 
    apex_ocr_task_satisfy_dependence(edtGuid, satisfyee, (int)location);
}



void traceTaskExecute(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                      ocrEdt_t fctPtr) {
    apex_ocr_task_execute(edtGuid, (apex_function_address)fctPtr, (int)location); 
}



void traceTaskFinish(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid) {
    apex_ocr_task_finish(edtGuid, (int)location);
 }

void traceTaskShift(u64 location, bool evtType, ocrTraceType_t objType,
                                            ocrTraceAction_t actionType, u64 workerId,
                                            u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                                            ocrEdt_t fctPtr, bool shiftFrom){
    if(shiftFrom) {
        apex_ocr_task_yield(edtGuid, (apex_function_address)fctPtr, (int)location);
    } else {
        apex_ocr_task_execute(edtGuid, (apex_function_address)fctPtr, (int)location);
    }

}


void traceTaskDataAcquire(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) { 
    apex_ocr_task_data_acquire(edtGuid, dbGuid, dbSize, (int)location);
}



void traceTaskDataRelease(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) {
    apex_ocr_task_data_release(edtGuid, dbGuid, dbSize, (int)location);
}



void traceEventCreate(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) {
    apex_ocr_event_create(eventGuid, (int)location);
}



void traceEventDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) {
    apex_ocr_event_destroy(eventGuid, (int)location);
}



void traceEventSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                 ocrTraceAction_t actionType, u64 workerId,
                                 u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid,
                                 ocrGuid_t satisfyee) {
    apex_ocr_event_satisfy_dependence(eventGuid, satisfyee, (int)location);
}



void traceEventAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                             ocrTraceAction_t actionType, u64 workerId,
                             u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                             ocrGuid_t dest) {
    apex_ocr_event_add_dependence(src, dest, (int)location);
}




void traceDataCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid,
                     u64 dbSize) { 
    apex_ocr_data_create(dbGuid, dbSize, (int)location);
}


void traceDataDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid) {
    apex_ocr_data_destroy(dbGuid, (int)location);
}

void platformSpecificInit(void * ocrConfig) {
    void(*original_platformSpecificInit)(void *);
    original_platformSpecificInit = (void (*)(void *))dlsym(RTLD_NEXT, "platformSpecificInit");
    (*original_platformSpecificInit)(ocrConfig);
    apex_ocr_init();
}

void platformSpecificFinalizer(u8 returnCode) {
    apex_ocr_shutdown();
    void(*original_platformSpecificFinalizer)(u8);
    original_platformSpecificFinalizer = (void (*)(u8))dlsym(RTLD_NEXT, "platformSpecificFinalizer");
    (*original_platformSpecificFinalizer)(returnCode);
}

void * hcRunWorker(void * worker) {
    void * (*original_hcRunWorker)(void *);
    original_hcRunWorker = (void * (*)(void *))dlsym(RTLD_NEXT, "hcRunWorker");
    void * result = (*original_hcRunWorker)(worker);
    apex_ocr_exit_thread();
    return result;
}


} // END extern "C"

#endif //apEX_HAVE_OCR
