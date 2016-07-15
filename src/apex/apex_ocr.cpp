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

static inline bool operator<(const ocrGuid_t & left, const ocrGuid_t & right) {
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
//static APEX_NATIVE_TLS ocrGuid_t current_guid = NULL_GUID;
static APEX_NATIVE_TLS bool thread_seen = false;
static std::atomic_bool pd_seen{false};

static inline guid_map_t * get_guid_map() {
    if(guid_map == nullptr) {
        guid_map = new std::map<ocrGuid_t, apex::profiler*>();
    }
    return guid_map;
}

static inline void apex_ocr_init() {
    std::cerr << "******** INIT ********" << std::endl;
    apex::init("main");
}

static inline void apex_ocr_shutdown() {
    std::cerr << "******** SHUTDOWN ********" << std::endl;
    apex::finalize();
}

static inline void apex_ocr_start(ocrGuid_t edtGuid, apex_function_address fctPtr, int node) {
    if(!thread_seen) {
        apex::register_thread("worker");
        thread_seen = true;
        if(!pd_seen) {
            pd_seen = true;
            apex::set_node_id(node);
        }
    }
    //current_guid = edtGuid;
    (*get_guid_map())[edtGuid] = apex::start(fctPtr);  
}

static inline void apex_ocr_stop(ocrGuid_t edtGuid) {
    //current_guid = NULL_GUID;
    apex::stop((*guid_map)[edtGuid]);
}

static inline void apex_ocr_new_task(apex_function_address fctPtr) {
    apex::new_task(fctPtr, nullptr);    
}

extern "C" {

void traceTaskCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                     ocrEdt_t fctPtr) { 
        apex_ocr_new_task((apex_function_address)fctPtr);
}



void traceTaskDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid) { }



void traceTaskRunnable(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid) { }



void traceTaskAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                            ocrTraceAction_t actionType, u64 workerId,
                            u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                            ocrGuid_t dest) {                                                                    
}



void traceTaskSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                ocrTraceAction_t actionType, u64 workerId,
                                u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                                ocrGuid_t satisfyee) { 
}



void traceTaskExecute(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                      ocrEdt_t fctPtr) {
    apex_ocr_start(edtGuid, (apex_function_address)fctPtr, (int)location); 
}



void traceTaskFinish(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid) {
    apex_ocr_stop(edtGuid);
 }



void traceTaskDataAcquire(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) { }



void traceTaskDataRelease(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                          ocrGuid_t dbGuid, u64 dbSize) { }



void traceEventCreate(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) { }



void traceEventDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                       ocrTraceAction_t actionType, u64 workerId,
                       u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid) { }



void traceEventSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                 ocrTraceAction_t actionType, u64 workerId,
                                 u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid,
                                 ocrGuid_t satisfyee) {
}



void traceEventAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                             ocrTraceAction_t actionType, u64 workerId,
                             u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                             ocrGuid_t dest) { 
}




void traceDataCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid,
                     u64 dbSize) { 
}


void traceDataDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid) { }

void platformSpecificInit(void * ocrConfig) {
    apex_ocr_init();
    void(*original_platformSpecificInit)(void *);
    original_platformSpecificInit = (void (*)(void *))dlsym(RTLD_NEXT, "platformSpecificInit");
    (*original_platformSpecificInit)(ocrConfig);
}

void platformSpecificFinalizer(u8 returnCode) {
    apex_ocr_shutdown();
    void(*original_platformSpecificFinalizer)(u8);
    original_platformSpecificFinalizer = (void (*)(u8))dlsym(RTLD_NEXT, "platformSpecificFinalizer");
    (*original_platformSpecificFinalizer)(returnCode);
}


} // END extern "C"

#endif //APEX_HAVE_OCR
