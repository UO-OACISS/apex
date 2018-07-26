#include <ompt.h>
#include <unordered_map>
#include "string.h"
#include "stdio.h"
#include "apex_api.hpp"
#include "apex_types.h"
#include "thread_instance.hpp"
#include "apex_cxx_shared_lock.hpp"
#include <atomic>

typedef enum apex_ompt_thread_type_e {
 apex_ompt_thread_initial = 1,
 apex_ompt_thread_worker = 2,
 apex_ompt_thread_other = 3
} apex_ompt_thread_type_t;

std::mutex threadid_mutex;
std::atomic<uint64_t> numthreads(0);
APEX_NATIVE_TLS uint64_t threadid(-1);

class linked_timer {
    public:
        void * prev;
        std::shared_ptr<apex::task_wrapper> tw;
        bool timing;
        inline void start(void) { apex::start(tw); timing = true;  }
        inline void yield(void) { apex::yield(tw); timing = false; }
        inline void stop(void)  { apex::stop(tw);  timing = false; }
        /* constructor */
        linked_timer(const char * name, 
            uint64_t task_id,
            void *p, 
            std::shared_ptr<apex::task_wrapper> &parent,
            bool auto_start) :
            prev(p), timing(auto_start) { 
            tw = apex::new_task(name, task_id, parent);
            if (auto_start) { this->start(); }
        }
        /* destructor */
        ~linked_timer() {
            if (timing) {
                apex::stop(tw);
            }
        }
};

/* Function pointers.  These are all queried from the runtime during
 *  * ompt_initialize() */
static ompt_set_callback_t ompt_set_callback;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

/* These methods are some helper functions for starting/stopping timers */

void apex_ompt_start(const char * state, ompt_data_t * ompt_data,
        ompt_data_t * region_data, bool auto_start) {
    static std::shared_ptr<apex::task_wrapper> nothing(nullptr);
    /* if the ompt_data->ptr pointer is not null, that means we have an implicit
     * parent and there's no need to specify a parent */
    linked_timer* tmp;
    if (ompt_data->ptr == nullptr && region_data != nullptr) {
        /* Get the parent scoped timer */
        linked_timer* parent = (linked_timer*)(region_data->ptr);
        if (parent != nullptr) {
            tmp = new linked_timer(state, ompt_data->value, ompt_data->ptr, parent->tw, auto_start);
        } else {
            tmp = new linked_timer(state, ompt_data->value, ompt_data->ptr, nothing, auto_start);
        }
#if 1
    } else if (ompt_data->ptr != nullptr) {
        /* Get the parent scoped timer */
        linked_timer* parent = (linked_timer*)(ompt_data->ptr);
        tmp = new linked_timer(state, ompt_data->value, ompt_data->ptr, parent->tw, true);
#endif
    } else {
        tmp = new linked_timer(state, ompt_data->value, ompt_data->ptr, nothing, auto_start);
    }

    /* Save the address of the scoped timer with the parallel region
     * or task, so we can stop the timer later */
    ompt_data->ptr = (void*)(tmp);
}

void apex_ompt_stop(ompt_data_t * ompt_data) {
    assert(ompt_data->ptr);
    void* tmp = ((linked_timer*)(ompt_data->ptr))->prev;
    delete((linked_timer*)(ompt_data->ptr));
    ompt_data->ptr = tmp;
}

/*
 * Mandatory Events
 * 
 * The following events are supported by all OMPT implementations.
 */

/* Event #1, thread begin */
extern "C" void apex_thread_begin(
    ompt_thread_type_t thread_type,       /* type of thread                      */
    ompt_data_t *thread_data              /* data of thread                      */
) {
    {
        std::unique_lock<std::mutex> l(threadid_mutex);
        threadid = numthreads++;
    }
    switch (thread_type) {
        case ompt_thread_initial:
            apex::register_thread("OpenMP Initial Thread");
            apex::sample_value("OpenMP Initial Thread", 1);
            break;
        case ompt_thread_worker:
            apex::register_thread("OpenMP Worker Thread");
            apex::sample_value("OpenMP Worker Thread", 1);
            break;
        case ompt_thread_other:
            apex::register_thread("OpenMP Other Thread");
            apex::sample_value("OpenMP Other Thread", 1);
            break;
        case ompt_thread_unknown:
        default:
            apex::register_thread("OpenMP Unknown Thread");
            apex::sample_value("OpenMP Unknown Thread", 1);
    }
}

/* Event #2, thread end */
extern "C" void apex_thread_end(
    ompt_data_t *thread_data              /* data of thread                      */
) {
    APEX_UNUSED(thread_data);
    apex::exit_thread();
}

/* Event #3, parallel region begin */
static void apex_parallel_region_begin (
    ompt_data_t *encountering_task_data,         /* data of encountering task           */
    const omp_frame_t *encountering_task_frame,  /* frame data of encountering task     */
    ompt_data_t *parallel_data,                  /* data of parallel region             */
    unsigned int requested_team_size,            /* requested number of threads in team */
    ompt_invoker_t invoker,                      /* invoker of master task              */
    const void *codeptr_ra                       /* return address of runtime call      */
) {
    char regionIDstr[128] = {0}; 
    sprintf(regionIDstr, "OpenMP Parallel Region: UNRESOLVED ADDR %p", codeptr_ra);
    apex_ompt_start(regionIDstr, parallel_data, NULL, true);
    //printf("%llu: Parallel Region Begin parent: %p, apex_parent: %p, region: %p, apex_region: %p\n", threadid, encountering_task_data, encountering_task_data->ptr, parallel_data, parallel_data->ptr); fflush(stdout);
}

/* Event #4, parallel region end */
static void apex_parallel_region_end (
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *encountering_task_data,  /* data of encountering task           */
    ompt_invoker_t invoker,               /* invoker of master task              */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    //printf("%llu: Parallel Region End parent: %p, apex_parent: %p, region: %p, apex_region: %p\n", threadid, encountering_task_data, encountering_task_data->ptr, parallel_data, parallel_data->ptr); fflush(stdout);
    apex_ompt_stop(parallel_data);
}

/* Event #5, task create */
extern "C" void apex_task_create (
    ompt_data_t *encountering_task_data,         /* data of parent task                 */
    const omp_frame_t *encountering_task_frame,  /* frame data for parent task          */
    ompt_data_t *new_task_data,                  /* data of created task                */
    ompt_task_type_t type,                       /* type of created task                */
    int has_dependences,                         /* created task has dependences        */
    const void *codeptr_ra                       /* return address of runtime call      */
) {
    char * type_str;
    static const char * initial_str = "OpenMP Initial Task";
    static const char * implicit_str = "OpenMP Implicit Task";
    static const char * explicit_str = "OpenMP Explicit Task";
    static const char * target_str = "OpenMP Target Task";
    static const char * undeferred_str = "OpenMP Undeferred Task";
    static const char * untied_str = "OpenMP Untied Task";
    static const char * final_str = "OpenMP Final Task";
    static const char * mergable_str = "OpenMP Mergable Task";
    static const char * merged_str = "OpenMP Merged Task";
    switch (type) {
        case ompt_task_initial:
            type_str = const_cast<char*>(initial_str);
            break;
        case ompt_task_implicit:
            type_str = const_cast<char*>(implicit_str);
            break;
        case ompt_task_explicit:
            type_str = const_cast<char*>(explicit_str);
            break;
        case ompt_task_target:
            type_str = const_cast<char*>(target_str);
            break;
        case ompt_task_undeferred:
            type_str = const_cast<char*>(undeferred_str);
            break;
        case ompt_task_untied:
            type_str = const_cast<char*>(untied_str);
            break;
        case ompt_task_final:
            type_str = const_cast<char*>(final_str);
            break;
        case ompt_task_mergeable:
            type_str = const_cast<char*>(mergable_str);
            break;
        case ompt_task_merged:
        default:
            type_str = const_cast<char*>(merged_str);
    }
    //printf("%llu: %s Task Create parent: %p, child: %p\n", threadid, type_str, encountering_task_data, new_task_data); fflush(stdout);

    if (codeptr_ra != NULL) {
        char regionIDstr[128] = {0}; 
        sprintf(regionIDstr, "%s: UNRESOLVED ADDR %p", type_str, codeptr_ra);
        apex_ompt_start(regionIDstr, new_task_data, encountering_task_data, false);
    } else {
        apex_ompt_start(type_str, new_task_data, encountering_task_data, false);
    }
}
 
/* Event #6, task schedule */
extern "C" void apex_task_schedule(
    ompt_data_t *prior_task_data,         /* data of prior task                  */
    ompt_task_status_t prior_task_status, /* status of prior task                */
    ompt_data_t *next_task_data           /* data of next task                   */
    ) {
    //printf("%llu: Task Schedule prior: %p, status: %d, next: %p\n", threadid, prior_task_data, prior_task_status, next_task_data); fflush(stdout);
    if (prior_task_data != nullptr) {
        linked_timer* prior = (linked_timer*)(prior_task_data->ptr);
        if (prior != nullptr) {
            switch (prior_task_status) {
                case ompt_task_yield:
                case ompt_task_others:
                    prior->yield();
                    break;
                case ompt_task_complete:
                case ompt_task_cancel:
                default:
                    void* tmp = prior->prev;
                    delete(prior);
                    prior_task_data->ptr = tmp;
            }
        }
    }
    //apex_ompt_start("OpenMP Task", next_task_data, NULL, true);
    linked_timer* next = (linked_timer*)(next_task_data->ptr);
    //assert(next);
    if (next != nullptr) {
        next->start();
    }
}

/* Event #7, implicit task */
extern "C" void apex_implicit_task(
    ompt_scope_endpoint_t endpoint,       /* endpoint of implicit task           */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of implicit task               */
    unsigned int team_size,               /* team size                           */
    unsigned int thread_num               /* thread number of calling thread     */
  ) {
    if (endpoint == ompt_scope_begin) {
        apex_ompt_start("OpenMP Implicit Task", task_data, parallel_data, false);
    } else {
        apex_ompt_stop(task_data);
    }
    //printf("%llu: Implicit Task task: %p, apex: %p, region: %p, %d\n", threadid, task_data, task_data->ptr, parallel_data, endpoint); fflush(stdout);
}

/* These are placeholder functions */

#if 0

/* Event #8, target */
extern "C" void apex_target (
    ompt_target_type_t kind,
    ompt_scope_endpoint_t endpoint,
    uint64_t device_num,
    ompt_data_t *task_data,
    ompt_id_t target_id,
    const void *codeptr_ra
) {
}

/* Event #9, target data */
extern "C" void apex_target_data_op (
    ompt_id_t target_id,
    ompt_id_t host_op_id,
    ompt_target_data_op_t optype,
    void *host_addr,
    void *device_addr,
    size_t bytes
) {
}

/* Event #10, target submit */
extern "C" void apex_target_submit (
    ompt_id_t target_id,
    ompt_id_t host_op_id
) {
}

/* Event #11, tool control */
extern "C" void apex_control(
    uint64_t command,                     /* command of control call             */
    uint64_t modifier,                    /* modifier of control call            */
    void *arg,                            /* argument of control call            */
    const void *codeptr_ra                /* return address of runtime call      */
    ) {
}

/* Event #12, device initialize */
extern "C" void apex_device_initialize (
    uint64_t device_num,
    const char *type,
    ompt_device_t *device,
    ompt_function_lookup_t lookup,
    const char *documentation
) {
}

/* Event #13, device finalize */
extern "C" void apex_device_finalize (
    uint64_t device_num
) {
}

/* Event #14, device load */
extern "C" void apex_device_load_t (
    uint64_t device_num,
    const char * filename,
    int64_t offset_in_file,
    void * vma_in_file,
    size_t bytes,
    void * host_addr,
    void * device_addr,
    uint64_t module_id
) {
}

/* Event #15, device load */
extern "C" void apex_device_unload (
    uint64_t device_num,
    uint64_t module_id
) {
}

#endif // placeholder functions

/**********************************************************************/
/* End Mandatory Events */
/**********************************************************************/

/**********************************************************************/
/* Optional events */
/**********************************************************************/

/* Event #16, sync region wait       */
extern "C" void apex_sync_region_wait (
    ompt_sync_region_kind_t kind,         /* kind of sync region                 */
    ompt_scope_endpoint_t endpoint,       /* endpoint of sync region             */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of task                        */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    char * tmp_str;
    static const char * barrier_str = "Barrier Wait";
    static const char * task_str = "Task Wait";
    static const char * task_group_str = "Task Group Wait";
    if (kind == ompt_sync_region_barrier) {
        tmp_str = const_cast<char*>(barrier_str);
    } else if (kind == ompt_sync_region_taskwait) {
        tmp_str = const_cast<char*>(task_str);
    } else if (kind == ompt_sync_region_taskgroup) {
        tmp_str = const_cast<char*>(task_group_str);
    }
    if (endpoint == ompt_scope_begin) {
        char regionIDstr[128] = {0}; 
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP %s: UNRESOLVED ADDR %p", tmp_str, codeptr_ra);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        } else {
            sprintf(regionIDstr, "OpenMP %s", tmp_str);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        }
    } else {
        apex_ompt_stop(task_data);
    }
}

/* Event #20, task at work begin or end       */
extern "C" void apex_ompt_work (
    ompt_work_type_t wstype,              /* type of work region                 */
    ompt_scope_endpoint_t endpoint,       /* endpoint of work region             */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of task                        */
    uint64_t count,                       /* quantity of work                    */
    const void *codeptr_ra                /* return address of runtime call      */
    ) {

    char * tmp_str;
    static const char * sections_str = "Sections";
    static const char * single_executor_str = "Single Executor";
    static const char * single_other_str = "Single Other";
    static const char * workshare_str = "Workshare";
    static const char * distribute_str = "Distribute";
    static const char * taskgroup_str = "Taskloop";
    static const char * loop_str = "Loop";
    if (wstype == ompt_work_sections) {
        tmp_str = const_cast<char*>(sections_str);
    } else if (wstype == ompt_work_single_executor) {
        tmp_str = const_cast<char*>(single_executor_str);
    } else if (wstype == ompt_work_single_other) {
        tmp_str = const_cast<char*>(single_other_str);
    } else if (wstype == ompt_work_workshare) {
        tmp_str = const_cast<char*>(workshare_str);
    } else if (wstype == ompt_work_distribute) {
        tmp_str = const_cast<char*>(distribute_str);
    } else if (wstype == ompt_work_taskloop) {
        tmp_str = const_cast<char*>(taskgroup_str);
    } else {
        tmp_str = const_cast<char*>(loop_str);
    }
    if (endpoint == ompt_scope_begin) {
        char regionIDstr[128] = {0}; 
        //printf("%llu: %s Begin task: %p, region: %p\n", threadid, tmp_str, task_data, parallel_data); fflush(stdout);
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Work %s: UNRESOLVED ADDR %p", tmp_str, codeptr_ra);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        } else {
            sprintf(regionIDstr, "OpenMP Work %s", tmp_str);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        }
    } else {
        //printf("%llu: %s End task: %p, region: %p\n", threadid, tmp_str, task_data, parallel_data); fflush(stdout);
        apex_ompt_stop(task_data);
    }
}

/* Event #21, task at master begin or end       */
extern "C" void apex_ompt_master (
    ompt_scope_endpoint_t endpoint,       /* endpoint of master region           */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of task                        */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    if (endpoint == ompt_scope_begin) {
        if (codeptr_ra != NULL) {
            char regionIDstr[128] = {0}; 
            sprintf(regionIDstr, "OpenMP Master: UNRESOLVED ADDR %p", codeptr_ra);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        } else {
            apex_ompt_start("OpenMP Master", task_data, parallel_data, true);
        }
    } else {
        apex_ompt_stop(task_data);
    }
}

/* Event #23, sync region begin or end */
extern "C" void apex_ompt_sync_region (
    ompt_sync_region_kind_t kind,         /* kind of sync region                 */
    ompt_scope_endpoint_t endpoint,       /* endpoint of sync region             */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of task                        */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    char * tmp_str;
    static const char * barrier_str = "Barrier";
    static const char * task_str = "Task";
    static const char * task_group_str = "Task Group";
    if (kind == ompt_sync_region_barrier) {
        tmp_str = const_cast<char*>(barrier_str);
    } else if (kind == ompt_sync_region_taskwait) {
        tmp_str = const_cast<char*>(task_str);
    } else if (kind == ompt_sync_region_taskgroup) {
        tmp_str = const_cast<char*>(task_group_str);
    }
    if (endpoint == ompt_scope_begin) {
        char regionIDstr[128] = {0}; 
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP %s: UNRESOLVED ADDR %p", tmp_str, codeptr_ra);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        } else {
            sprintf(regionIDstr, "OpenMP %s", tmp_str);
            apex_ompt_start(regionIDstr, task_data, parallel_data, true);
        }
    } else {
        apex_ompt_stop(task_data);
    }
}

/* Event #29, flush event */
extern "C" void apex_ompt_flush (
    ompt_data_t *thread_data,             /* data of thread                      */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    if (codeptr_ra != NULL) {
        char regionIDstr[128] = {0}; 
        sprintf(regionIDstr, "OpenMP Flush: UNRESOLVED ADDR %p", codeptr_ra);
        apex::sample_value(regionIDstr, 1);
    } else {
        apex::sample_value(std::string("OpenMP Flush"),1);
    }
}

/* Event #30, cancel event */
extern "C" void apex_ompt_cancel (
    ompt_data_t *task_data,               /* data of task                        */
    int flags,                            /* cancel flags                        */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    char regionIDstr[128] = {0}; 
    if (flags & ompt_cancel_parallel) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Parallel: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Parallel"),1);
        }
    }
    if (flags & ompt_cancel_sections) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Sections: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Sections"),1);
        }
    }
    if (flags & ompt_cancel_do) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Do: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Do"),1);
        }
    }
    if (flags & ompt_cancel_taskgroup) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Taskgroup: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Taskgroup"),1);
        }
    }
    if (flags & ompt_cancel_activated) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Activated: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Activated"),1);
        }
    }
    if (flags & ompt_cancel_detected) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Detected: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Detected"),1);
        }
    }
    if (flags & ompt_cancel_discarded_task) {
        if (codeptr_ra != NULL) {
            sprintf(regionIDstr, "OpenMP Cancel Discarded Task: UNRESOLVED ADDR %p", codeptr_ra);
            apex::sample_value(std::string(regionIDstr),1);
        } else {
            apex::sample_value(std::string("OpenMP Cancel Discarded Task"),1);
        }
    }
    apex_ompt_stop(task_data);
}

/* Event #30, cancel event */
extern "C" void apex_ompt_idle (
    ompt_scope_endpoint_t endpoint        /* endpoint of idle time               */
) {
    static APEX_NATIVE_TLS apex::profiler* p = nullptr;
    if (endpoint == ompt_scope_begin) {
        p = apex::start("OpenMP Idle");
    } else {
        apex::stop(p);
    }
}

/**********************************************************************/
/* End Optional events */
/**********************************************************************/

// This function is for checking that the function registration worked.
int apex_ompt_register(ompt_callbacks_t e, ompt_callback_t c , const char * name) {
  fprintf(stderr,"Registering OMPT callback %s...",name); fflush(stderr);
  if (ompt_set_callback(e, c) == 0) { \
    fprintf(stderr,"\n\tFailed to register OMPT callback %s!\n",name); fflush(stderr);
  } else {
    fprintf(stderr,"success.\n");
  }
  return 0;
}

extern "C" {

int ompt_initialize(ompt_function_lookup_t lookup, ompt_data_t* tool_data) {
    {
        std::unique_lock<std::mutex> l(threadid_mutex);
        threadid = numthreads++;
    }
    fprintf(stderr,"Getting OMPT functions..."); fflush(stderr);
    ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
    ompt_get_task_info = (ompt_get_task_info_t) lookup("ompt_get_task_info");
    ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
    ompt_get_parallel_info = (ompt_get_parallel_info_t) lookup("ompt_get_parallel_info");
    ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");

    ompt_get_num_places = (ompt_get_num_places_t) lookup("ompt_get_num_places");
    ompt_get_place_proc_ids = (ompt_get_place_proc_ids_t) lookup("ompt_get_place_proc_ids");
    ompt_get_place_num = (ompt_get_place_num_t) lookup("ompt_get_place_num");
    ompt_get_partition_place_nums = (ompt_get_partition_place_nums_t) lookup("ompt_get_partition_place_nums");
    ompt_get_proc_id = (ompt_get_proc_id_t) lookup("ompt_get_proc_id");
    ompt_enumerate_states = (ompt_enumerate_states_t) lookup("ompt_enumerate_states");
    ompt_enumerate_mutex_impls = (ompt_enumerate_mutex_impls_t) lookup("ompt_enumerate_mutex_impls");

    apex::init("OpenMP Program",0,1);
    fprintf(stderr,"Registering OMPT events..."); fflush(stderr);

    /* Mandatory events */

    // Event 1: thread begin
    apex_ompt_register(ompt_callback_thread_begin,
        (ompt_callback_t)&apex_thread_begin, "thread_begin");
    // Event 2: thread end
    apex_ompt_register(ompt_callback_thread_end,
        (ompt_callback_t)&apex_thread_end, "thread_end");
    // Event 3: parallel begin
    apex_ompt_register(ompt_callback_parallel_begin,
        (ompt_callback_t)&apex_parallel_region_begin, "parallel_begin");
    // Event 4: parallel end
    apex_ompt_register(ompt_callback_parallel_end,
        (ompt_callback_t)&apex_parallel_region_end, "parallel_end");
    if (apex::apex_options::ompt_high_overhead_events()) {
        // Event 5: task create
        apex_ompt_register(ompt_callback_task_create,
            (ompt_callback_t)&apex_task_create, "task_create");
        // Event 6: task schedule (start/stop)
        apex_ompt_register(ompt_callback_task_schedule,
            (ompt_callback_t)&apex_task_schedule, "task_schedule");
/* The LLVM runtime seems to be not storing the implicit task data correctly.
 * Disable this event for now. */
#if !defined(__clang__)
        // Event 7: implicit task (start/stop)
        apex_ompt_register(ompt_callback_implicit_task,
            (ompt_callback_t)&apex_implicit_task, "implicit_task");
#endif
    }

 #if 0
    // Event 8: target
    apex_ompt_register(ompt_callback_target,
        (ompt_callback_t)&apex_target, "target");
    // Event 9: target data operation
    apex_ompt_register(ompt_callback_target_data_op,
        (ompt_callback_t)&apex_target_data_op, "target_data_operation");
    // Event 10: target submit
    apex_ompt_register(ompt_callback_target_submit,
        (ompt_callback_t)&apex_target_submit, "target_submit");
    // Event 11: control tool
    apex_ompt_register(ompt_callback_control_tool,
        (ompt_callback_t)&apex_control, "event_control");
    // Event 12: device initialize
    apex_ompt_register(ompt_callback_device_initialize,
        (ompt_callback_t)&apex_device_initialize, "device_initialize");
    // Event 13: device finalize
    apex_ompt_register(ompt_callback_device_finalize,
        (ompt_callback_t)&apex_device_finalize, "device_finalize");
    // Event 14: device load
    apex_ompt_register(ompt_callback_device_load,
        (ompt_callback_t)&apex_device_load, "device_load");
    // Event 15: device unload
    apex_ompt_register(ompt_callback_device_unload,
        (ompt_callback_t)&apex_device_unload, "device_unload");

#endif

    /* optional events */

    if (!apex::apex_options::ompt_required_events_only()) {
        // Event 20: task at work begin or end
        apex_ompt_register(ompt_callback_work,
            (ompt_callback_t)&apex_ompt_work, "work");
        /* Event 21: task at master begin or end     */
        apex_ompt_register(ompt_callback_master,
            (ompt_callback_t)&apex_ompt_master, "master");
#if 0
        /* Event 22: target map                      */
#endif
        /* Event 29: after executing flush           */
        apex_ompt_register(ompt_callback_flush,
            (ompt_callback_t)&apex_ompt_flush, "flush");
        /* Event 30: cancel innermost binding region */
        apex_ompt_register(ompt_callback_cancel,
            (ompt_callback_t)&apex_ompt_cancel, "cancel");

        if (apex::apex_options::ompt_high_overhead_events()) {
            // Event 16: sync region wait begin or end
            apex_ompt_register(ompt_callback_sync_region_wait,
                (ompt_callback_t)&apex_sync_region_wait, "sync_region_wait");
#if 0
            // Event 17: mutex released
            apex_ompt_register(ompt_callback_mutex_released,
                (ompt_callback_t)&apex_mutex_released, "mutex_released");
            // Event 18: report task dependences
            apex_ompt_register(ompt_callback_report_task_dependences,
                (ompt_callback_t)&apex_report_task_dependences, "mutex_report_task_dependences");
            // Event 19: report task dependence
            apex_ompt_register(ompt_callback_report_task_dependence,
                (ompt_callback_t)&apex_report_task_dependence, "mutex_report_task_dependence");
#endif
            /* Event 23: sync region begin or end        */
            apex_ompt_register(ompt_callback_sync_region,
                (ompt_callback_t)&apex_ompt_sync_region, "sync_region");
            /* Event 31: begin or end idle state         */
            apex_ompt_register(ompt_callback_idle,
                (ompt_callback_t)&apex_ompt_idle, "idle");
#if 0
            /* Event 24: lock init                       */
            /* Event 25: lock destroy                    */
            /* Event 26: mutex acquire                   */
            apex_ompt_register(ompt_callback_mutex_acquire,
                (ompt_callback_t)&apex_mutex_acquire, "mutex_acquire");
            /* Event 27: mutex acquired                  */
            apex_ompt_register(ompt_callback_mutex_acquired,
                (ompt_callback_t)&apex_mutex_acquired, "mutex_acquired");
            /* Event 28: nest lock                       */
#endif
        }

    }

    fprintf(stderr,"done.\n"); fflush(stderr);
    return 1;
}

void ompt_finalize(ompt_data_t* tool_data)
{
    printf("OpenMP runtime is shutting down...\n");
    apex::finalize();
}

ompt_start_tool_result_t * ompt_start_tool(
    unsigned int omp_version, const char *runtime_version) {
    static ompt_start_tool_result_t result;
    result.initialize = &ompt_initialize;
    result.finalize = &ompt_finalize;
    result.tool_data.value = 0L;
    result.tool_data.ptr = NULL;
    return &result;
}

}; // extern "C"
