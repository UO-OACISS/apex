#ifdef APEX_HAVE_OCR

#include <map>
#include <iostream>
#include <cinttypes>
#include <cstdio>
#include <atomic>
#include <utility>
#include <stack>
#include <sstream>
#include <string>
#include <limits>
#include <vector>
#include "apex_api.hpp"
#include "apex_types.h"
#include "apex.hpp"
#include "thread_instance.hpp"
#include <random>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include <mpi.h>

#define OCR_TYPE_H x86.h
#include "ocr.h"
#include "ocr-comm-platform.h"

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#ifdef APEX_OCR_DEBUG
#define DEBUG_MSG(str) std::cerr << str << std::endl
#else
#define DEBUG_MSG(str)
#endif

//#define APEX_OCR_LOAD_BALANCE

constexpr uint32_t MESSAGES_BETWEEN_LOAD_SENDS = 100;

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
static APEX_NATIVE_TLS ocrGuid_t last_guid = NULL_GUID;
static APEX_NATIVE_TLS apex::profiler * profiler_to_yield = nullptr;
static APEX_NATIVE_TLS ocrGuid_t guid_to_yield = NULL_GUID;
static std::atomic_bool pd_seen{false};
static std::atomic<uint64_t> thread_id{0};

static bool done = true;
static uint64_t ** loads = nullptr;
static MPI_Win * wins = nullptr;
static int my_rank = -1;
static int num_ranks = -1;
static uint32_t num_messages = MESSAGES_BETWEEN_LOAD_SENDS - 1;
static uint64_t version = 0;

static std::vector<double> * my_cdf = nullptr;

static inline int get_rand_rank() {
    static std::random_device seeder;
    static std::mt19937 engine(seeder());
    static std::uniform_int_distribution<int> dist(0, num_ranks-1);
    const int compGuess = dist(engine);
    return compGuess;
}

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
    apex::apex::instance()->set_runtime(APEX_RUNTIME_OCR);
    apex::register_periodic_policy(100000, [](apex_context const & context) {
        if(done || my_rank < 0 || num_ranks < 0) {
            return APEX_NOERROR;
        }
        // Update my row from the other rows
        uint64_t * my_row = loads[my_rank];
        for(int row = 0; row < num_ranks; ++row) {
            if(row == my_rank) {
                continue;
            }
            const uint64_t * remote_row = loads[row];
            for(int col = 0; col < num_ranks; ++col) {
                if(col == my_rank) {
                    continue;
                }
                const size_t col_offset = 4*col;
                const uint64_t my_version = my_row[col_offset];
                const uint64_t remote_version = remote_row[col_offset];
                //fprintf(stderr, "[%d] row %d col %d my version %llu rem version %llu\n", my_rank, row, col, my_version, remote_version);
                if(my_version < remote_version) {
                    //fprintf(stderr, "[%d] rank %d has a newer version for rank %d \n", my_rank, row, col);
                    my_row[col_offset]   =  remote_version;
                    my_row[col_offset+1] =  remote_row[col_offset+1];
                    my_row[col_offset+2] =  remote_row[col_offset+2];
                    my_row[col_offset+3] =  remote_row[col_offset+3];
                }
            }
        }
        // Update my CDF from my row
        uint64_t sum = 0;
        for(int col = 0; col < num_ranks; ++col) {
            const size_t col_offset = 4*col;
            sum += my_row[col_offset+2] + my_row[col_offset+3];
        }
        
        return APEX_NOERROR;
    });
}

static inline int apex_ocr_get_task_placement() {
    static const apex_load_balance_policy_t policy = apex::get_load_balance_policy();
    static int next_rank = 0;
    switch(policy) {
        case APEX_LOAD_BALANCE_NONE:
            return -1;
        case APEX_LOAD_BALANCE_RANDOM:
            return get_rand_rank();
        case APEX_LOAD_BALANCE_SINGLE:
            return 1;
        case APEX_LOAD_BALANCE_SAME:
            return my_rank;
        case APEX_LOAD_BALANCE_RR: {
            ++next_rank;
            if(next_rank == num_ranks) {
                next_rank = 0;
            }
            return next_rank;
        }
        case APEX_LOAD_BALANCE_LEAST: {
            uint64_t * my_row = loads[my_rank];
            int least_loaded = 0;
            int min_load = std::numeric_limits<int>::max();
            for(int col = 0; col < num_ranks; ++col) {
                const size_t col_offset = 4*col;
                const int load = my_row[col_offset+2] + my_row[col_offset+3];
                if(load < min_load) {
                    min_load = load;
                    least_loaded = col;
                }
            }
            return least_loaded;
        }
        case APEX_LOAD_BALANCE_PROB: {
            //static std::default_random_engine generator;
            //static std::uniform_real_distribution<double> distribution(0.0, 1.0);
            //constexpr double eps = 1e-9;
            //double r = distribution(generator);

            //for(int i = 0; i < num_ranks; ++i) {
            //    r -= my_cdf[i];
            //    if(r < eps) {
            //        return i;
            //    }
            //}
            //return num_ranks-1;
            PRINTF("PROB not implemented\n");
            return -1;
        }
        default:
            PRINTF("Load balance mode not supported\n");
    }
    return -1;
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
    PRINTF("APEX shutdown\n");
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
    // If we're coming back immediately to the same task,
    // then we yielded and immediately resumed without
    // actually shifting to another task.
    if(ocrGuidIsEq(edtGuid, last_guid)) {
        profiler_to_yield = nullptr;
        guid_to_yield = NULL_GUID;
        return;
    }
    // Do any deferred yield
    if(profiler_to_yield != nullptr) {
        apex::yield(profiler_to_yield);
        get_guid_map()->erase(guid_to_yield);
        profiler_to_yield = nullptr;
        guid_to_yield = NULL_GUID;
    }
    apex::task_identifier * task_id = new apex::task_identifier(fctPtr, guid_to_str(edtGuid));
    apex::profiler * p = apex::start(task_id);
    last_guid = edtGuid;
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
    profiler_to_yield = p;
    guid_to_yield = edtGuid;
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

static inline void apex_ocr_init_mpi() {
    const apex_load_balance_policy_t policy = apex::get_load_balance_policy();
    switch(policy) {
        case APEX_LOAD_BALANCE_NONE:
            PRINTF("~~~~ Load balance: NONE\n");
            break;
        case APEX_LOAD_BALANCE_RANDOM:
            PRINTF("~~~~ Load balance: RANDOM\n");
            break;
        case APEX_LOAD_BALANCE_SINGLE:
            PRINTF("~~~~ Load balance: SINGLE\n");
            break;
        case APEX_LOAD_BALANCE_RR: 
            PRINTF("~~~~ Load balance: RR\n");
            break;
        case APEX_LOAD_BALANCE_LEAST:
            PRINTF("~~~~ Load balance: LEAST\n");
            break;
        case APEX_LOAD_BALANCE_PROB:
            PRINTF("~~~~ Load balance: PROB\n");
            break;
        default:
            PRINTF("~~~~ Load balance: UNKNOWN????\n");
    }
    version = 0;
    num_messages = MESSAGES_BETWEEN_LOAD_SENDS - 1;
    done = false;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    my_cdf = new std::vector<double>(num_ranks);
    // one "loads" array per rank
    MPI_Alloc_mem(num_ranks * sizeof(uint64_t *), MPI_INFO_NULL, &loads);
    // one window per rank
    MPI_Alloc_mem(num_ranks * sizeof(MPI_Win), MPI_INFO_NULL, &wins);
    for(int i = 0; i < num_ranks; ++i) {
        const MPI_Aint mem_size = num_ranks * 4 * sizeof(uint64_t);
        MPI_Alloc_mem(mem_size, MPI_INFO_NULL, &(loads[i]));    
        memset(loads[i], 0, mem_size);
        MPI_Win_create(loads[i], mem_size, sizeof(uint64_t), MPI_INFO_NULL, MPI_COMM_WORLD, wins+i);
    }
}

static inline void apex_ocr_finalize_mpi() {
    PRINTF("MPI finalize\n");
    done = true;
    for(int i = 0; i < num_ranks; ++i) {
        MPI_Win_free(wins+i);
        MPI_Free_mem(loads[i]);
    }
    MPI_Free_mem(loads);
    delete my_cdf;
}

static inline void apex_ocr_send_load(int dest) {
    // update my own row (loads FROM my rank)
    ++version;
    int created = apex::get_local_tasks_created();
    int eligible = apex::get_local_tasks_eligible();
    int running = apex::get_local_tasks_running();
    //if(created < 0 || eligible < 0 || running < 0) {
    //    fprintf(stderr, "[%d] ERROR: bad number, ver=%d created=%d eligible=%d running=%d\n", my_rank, version, created, eligible, running);
    //    MPI_Abort(MPI_COMM_WORLD, 1);
    //    return;
    //}
    const size_t my_offset = 4 * my_rank;
    uint64_t * my_row = loads[my_rank];
    my_row[my_offset] = version;
    my_row[my_offset+1] = created;
    my_row[my_offset+2] = eligible;
    my_row[my_offset+3] = running;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, dest, 0, wins[my_rank]);
    MPI_Put(my_row, 4*num_ranks, MPI_UINT64_T, dest, 0, 4*num_ranks, MPI_UINT64_T, wins[my_rank]);
    MPI_Win_unlock(dest, wins[my_rank]);
    if(num_ranks > 2) {
        int other_rank = my_rank;
        while(other_rank == my_rank) {
            other_rank = get_rand_rank();
        }
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, other_rank, 0, wins[my_rank]);
        MPI_Put(my_row, 4*num_ranks, MPI_UINT64_T, other_rank, 0, 4*num_ranks, MPI_UINT64_T, wins[my_rank]);
        MPI_Win_unlock(other_rank, wins[my_rank]);
    }
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

void traceTemplateCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t templGuid,
                     ocrEdt_t fctPtr) {
    return;
}

void traceAPITemplateCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrEdt_t fctPtr,
                     u32 paramc, u32 depc) {
    return;
}

void traceAPITaskCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t edtGuid,
                     ocrGuid_t templGuid, u32 paramc, u64 * paramv,
                     u32 depc, ocrGuid_t * depv) {
    //PRINTF("API task create: " GUIDF "\n", GUIDA(edtGuid));
    return;
}

void traceAPIAddDependence(u64 location, bool evtType, ocrTraceType_t objType,
                            ocrTraceAction_t actionType, u64 workerId,
                            u64 timestamp, ocrGuid_t parent, ocrGuid_t src,
                            ocrGuid_t dest, u32 slot, ocrDbAccessMode_t mode) {
    return;
}

void traceAPIEventSatisfyDependence(u64 location, bool evtType, ocrTraceType_t objType,
                                 ocrTraceAction_t actionType, u64 workerId,
                                 u64 timestamp, ocrGuid_t parent, ocrGuid_t eventGuid,
                                 ocrGuid_t satisfyee, u32 slot) {
    return;
}

void traceAPIEventCreate(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrEventTypes_t eventType) {
    return;
}

void traceAPIDataCreate(u64 location, bool evtType, ocrTraceType_t objType,
                     ocrTraceAction_t actionType, u64 workerId,
                     u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid,
                     u64 dbSize) {
    return;
}

void traceAPIDataDestroy(u64 location, bool evtType, ocrTraceType_t objType,
                      ocrTraceAction_t actionType, u64 workerId,
                      u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid) {
    return;
}

void traceAPIDataRelease(u64 location, bool evtType, ocrTraceType_t objType,
                          ocrTraceAction_t actionType, u64 workerId,
                          u64 timestamp, ocrGuid_t parent, ocrGuid_t dbGuid) {
    return;
}

#ifdef APEX_OCR_LOAD_BALANCE

u8 getTaskPlacement(ocrGuid_t edtGuid, u32 * location) {
    const int proposed_location = apex_ocr_get_task_placement();
    if(proposed_location < 0) {
        return 1;
    }
    *location = proposed_location;
    PRINTF("APEX proposing move to %d\n", proposed_location);
    return 0; 
}

void platformInitMPIComm(int * argc, char *** argv) {
    void(*original_platformInitMPIComm)(int *, char ***);
    original_platformInitMPIComm = (void (*)(int *, char ***))dlsym(RTLD_NEXT, "platformInitMPIComm");
    if(original_platformInitMPIComm == nullptr) {
        fprintf(stderr, "OCR wrapper: platformInitMPIComm not found.\n");
        abort();
    }
    (*original_platformInitMPIComm)(argc, argv);
    apex_ocr_init_mpi();
}

void platformFinalizeMPIComm() {
    apex_ocr_finalize_mpi();
    void(*original_platformFinalizeMPIComm)();
    original_platformFinalizeMPIComm = (void (*)())dlsym(RTLD_NEXT, "platformFinalizeMPIComm");
    if(original_platformFinalizeMPIComm == nullptr) {
        fprintf(stderr, "OCR wrapper: original_platformFinalizeMPIComm not found.\n");
        abort();
    }
    (*original_platformFinalizeMPIComm)();
}

static u8 (*original_sendMessage)(struct _ocrCommPlatform_t* self, ocrLocation_t target, struct _ocrPolicyMsg_t *message, u64 *id, u32 properties, u32 mask) = nullptr;

static u8 wrapped_sendMessage(struct _ocrCommPlatform_t* self, ocrLocation_t target, struct _ocrPolicyMsg_t *message, u64 *id, u32 properties, u32 mask) {
    ++num_messages;
    //int created = apex::get_local_tasks_created();
    //int eligible = apex::get_local_tasks_eligible();
    //int running = apex::get_local_tasks_running();
    //fprintf(stderr, "[%d] created=%d eligible=%d running=%d\n", my_rank, created, eligible, running);
    if(num_messages == MESSAGES_BETWEEN_LOAD_SENDS) {
        num_messages = 0;
        // Send load information
        apex_ocr_send_load(target);
        //if(num_messages % 10000 == 0) {
        //    fprintf(stderr, "[%d] ", my_rank);
        //    for(int i = 0; i < num_ranks; ++i) {
        //        const size_t offset = i*4;
        //        const uint64_t * my_loads = loads[my_rank];
        //        fprintf(stderr, "(%llu %llu %llu %llu) ", my_loads[offset], my_loads[offset+1], my_loads[offset+2], my_loads[offset+3]);
        //    }
        //    fprintf(stderr, "\n");
        //}
    }
    return original_sendMessage(self, target, message, id, properties, mask);
}

ocrCommPlatformFactory_t *newCommPlatformFactoryMPI(ocrParamList_t *perType) {
    ocrCommPlatformFactory_t * (*original_newCommPlatformFactoryMPI)(ocrParamList_t *);
    original_newCommPlatformFactoryMPI = (ocrCommPlatformFactory_t * (*)(ocrParamList_t *))dlsym(RTLD_NEXT, "newCommPlatformFactoryMPI");
    if(original_newCommPlatformFactoryMPI == nullptr) {
        fprintf(stderr, "OCR wrapper: original_newCommPlatformFactoryMPI not found.\n");
        abort();
    }
    ocrCommPlatformFactory_t * result = (*original_newCommPlatformFactoryMPI)(perType);
    original_sendMessage = result->platformFcts.sendMessage;
    result->platformFcts.sendMessage = wrapped_sendMessage;
    return result;
}

#endif // APEX_OCR_LOAD_BALANCE

void platformSpecificInit(void * ocrConfig) {
    apex_ocr_init();
    void(*original_platformSpecificInit)(void *);
    original_platformSpecificInit = (void (*)(void *))dlsym(RTLD_NEXT, "platformSpecificInit");
    if(original_platformSpecificInit == nullptr) {
        fprintf(stderr, "OCR wrapper: original_platformSpecificInit not found.\n");
        abort();
    }
    (*original_platformSpecificInit)(ocrConfig);
}

void platformSpecificFinalizer(u8 returnCode) {
    apex_ocr_shutdown();
    void(*original_platformSpecificFinalizer)(u8);
    original_platformSpecificFinalizer = (void (*)(u8))dlsym(RTLD_NEXT, "platformSpecificFinalizer");
    if(original_platformSpecificFinalizer == nullptr) {
        fprintf(stderr, "OCR wrapper: original_platformSpecificFinalizer not found.\n");
        abort();
    }
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
