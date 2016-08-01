#ifdef APEX_HAVE_OCR

#include <map>
#include <iostream>
#include <cinttypes>
#include <cstdio>
#include <atomic>
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
static APEX_NATIVE_TLS ocrGuid_t current_guid = NULL_GUID;
static APEX_NATIVE_TLS bool thread_seen = false;
static std::atomic_bool pd_seen{false};

static inline guid_map_t * get_guid_map() {
    if(unlikely(guid_map == nullptr)) {
        guid_map = new std::map<ocrGuid_t, apex::profiler*>();
    }
    return guid_map;
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

static inline void apex_ocr_task_create(ocrGuid_t edtGuid, apex_function_address fctPtr) {
    DEBUG_MSG("Task created: " << guid_to_str(edtGuid));
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::new_task(task_id, nullptr);    
} 

static inline void apex_ocr_task_destroy(ocrGuid_t edtGuid, apex_function_address fctPtr) {
    DEBUG_MSG("Task destroyed: " << guid_to_str(edtGuid));
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::destroy_task(task_id, nullptr);
}


static inline void apex_ocr_task_runnable(ocrGuid_t edtGuid) {
    DEBUG_MSG("Task runnable: " << guid_to_str(edtGuid));
}

static inline void apex_ocr_task_add_dependence(ocrGuid_t src, ocrGuid_t dest) {
    DEBUG_MSG("Task add dependence: " << guid_to_str(src) << " --> " << guid_to_str(dest));
}


static inline void apex_ocr_task_satisfy_dependence(ocrGuid_t edtGuid, ocrGuid_t satisfyee) {
    DEBUG_MSG("Task satisfy dependence. Task: " << guid_to_str(edtGuid) << " Satisfyee: " << guid_to_str(satisfyee));
}

static inline void apex_ocr_task_execute(ocrGuid_t edtGuid, apex_function_address fctPtr, int node) {
    DEBUG_MSG("Task executing: " << guid_to_str(edtGuid));
    if(unlikely(!thread_seen)) {
        apex::register_thread("worker");
        thread_seen = true;
        if(unlikely(!pd_seen)) {
            pd_seen = true;
            apex::set_node_id(node);
        }
    }
    current_guid = edtGuid;
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    (*get_guid_map())[edtGuid] = apex::start(task_id);  
}

static inline void apex_ocr_task_finish(ocrGuid_t edtGuid) {
    DEBUG_MSG("Task finished: " << guid_to_str(edtGuid));
    current_guid = NULL_GUID;
    apex::stop((*guid_map)[edtGuid]);
}


static inline void apex_ocr_task_data_acquire(ocrGuid_t edtGuid, ocrGuid_t dbGuid, u64 dbSize) {
    DEBUG_MSG("Task data acquire. Task " << guid_to_str(edtGuid) << " acquired DB " << guid_to_str(dbGuid) << " of size " << dbSize);
}

static inline void apex_ocr_task_data_release(ocrGuid_t edtGuid, ocrGuid_t dbGuid, u64 dbSize) {
    DEBUG_MSG("Task data release. Task " << guid_to_str(edtGuid) << " released DB " << guid_to_str(dbGuid) << " of size " << dbSize);
}

static inline void apex_ocr_event_create(ocrGuid_t eventGuid) {
    DEBUG_MSG("Event create: " << guid_to_str(eventGuid));
}

static inline void apex_ocr_event_destroy(ocrGuid_t eventGuid) {
    DEBUG_MSG("Event destroy: " << guid_to_str(eventGuid));
}

static inline void apex_ocr_event_satisfy_dependence(ocrGuid_t eventGuid, ocrGuid_t satisfyee) {
    DEBUG_MSG("Event satisfy dependence. Event: " << guid_to_str(eventGuid) << " Satisfyee: " << guid_to_str(satisfyee));
}

static inline void apex_ocr_event_add_dependence(ocrGuid_t src, ocrGuid_t dest) {
    DEBUG_MSG("Event add dependence: " << guid_to_str(src) << " --> " << guid_to_str(dest));
}

static inline void apex_ocr_data_create(ocrGuid_t dbGuid, u64 dbSize) {
    DEBUG_MSG("Data create: " << guid_to_str(dbGuid) << " of size " << dbSize);
}

static inline void apex_ocr_data_destroy(ocrGuid_t dbGuid) {
    DEBUG_MSG("Data destroy: " << guid_to_str(dbGuid));
}

extern "C" {

void traceTaskCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                     ocrEdt_t fctPtr) { 
    apex_ocr_task_create(edtGuid, (apex_function_address)fctPtr);
}



void traceTaskDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                      ocrEdt_t fctPtr) {
    apex_ocr_task_destroy(edtGuid, (apex_function_address)fctPtr);
}



void traceTaskRunnable(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid) { 
    apex_ocr_task_runnable(edtGuid);
}



void traceTaskAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                            ocrTraceAction_t actionType, u64 workerId,
                            u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                            ocrGuid_t dest) {                                                                    
    apex_ocr_task_add_dependence(src, dest);
}



void traceTaskSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                ocrTraceAction_t actionType, u64 workerId,
                                u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                                ocrGuid_t satisfyee) { 
    apex_ocr_task_satisfy_dependence(edtGuid, satisfyee);
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
    apex_ocr_task_finish(edtGuid);
 }



void traceTaskDataAcquire(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) { 
    apex_ocr_task_data_acquire(edtGuid, dbGuid, dbSize);
}



void traceTaskDataRelease(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) {
    apex_ocr_task_data_release(edtGuid, dbGuid, dbSize);
}



void traceEventCreate(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) {
    apex_ocr_event_create(eventGuid);
}



void traceEventDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) {
    apex_ocr_event_destroy(eventGuid);
}



void traceEventSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                 ocrTraceAction_t actionType, u64 workerId,
                                 u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid,
                                 ocrGuid_t satisfyee) {
    apex_ocr_event_satisfy_dependence(eventGuid, satisfyee);
}



void traceEventAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                             ocrTraceAction_t actionType, u64 workerId,
                             u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                             ocrGuid_t dest) {
    apex_ocr_event_add_dependence(src, dest);
}




void traceDataCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid,
                     u64 dbSize) { 
    apex_ocr_data_create(dbGuid, dbSize);
}


void traceDataDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid) {
    apex_ocr_data_destroy(dbGuid);
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

#endif //APEX_HAVE_OCR
