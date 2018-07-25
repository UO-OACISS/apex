#include <ompt.h>
#include <unordered_map>
#include <stack>
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

APEX_NATIVE_TLS std::stack<std::shared_ptr<apex::task_wrapper> > *timer_stack;

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

void apex_ompt_task_start(const char * state, ompt_data_t * task_id) {
  std::shared_ptr<apex::task_wrapper> tmp = apex::new_task(std::string(state), task_id->value);
  // Save the address of the shared pointer with the task, so we can stop
  // the timer later
  task_id->ptr = (void*)(&tmp);
  // start the APEX timer
  apex::start(tmp);
  // unfortunately, we need to store a local reference to the timer so that it
  // doesn't go out of scope and get destroyed. This stack may get replaced with
  // a hash map, using the ID of the task.
  timer_stack->push(std::move(tmp));
}

void apex_ompt_task_stop(const char * state, ompt_data_t * task_id) {
  // ideally, we would use the timer that we stored with the task_id
  // structure.  However, the LLVM runtime isn't maintaining it for us,
  // so we will *assume* that this thread started this task, and stop
  // its top level timer. This could be a bad assumption with the HPX
  // OpenMP implementation.
  if (timer_stack->empty()) { // uh-oh...
    apex::profiler * p = nullptr;
    apex::stop(p);
  } else {
    std::shared_ptr<apex::task_wrapper>& p = timer_stack->top();
    apex::stop(p);
    timer_stack->pop();
  }
}

void apex_ompt_start(const char * state, ompt_data_t * parallel_id) {
  std::shared_ptr<apex::task_wrapper> tmp = apex::new_task(std::string(state), parallel_id->value);
  parallel_id->ptr = (void*)(&tmp);
  apex::start(tmp);
  timer_stack->push(std::move(tmp));
}

void apex_ompt_stop(const char * state, ompt_data_t * parallel_id) {
  std::shared_ptr<apex::task_wrapper>& p = timer_stack->top();
  apex::stop(p);
  timer_stack->pop();
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
    APEX_UNUSED(thread_type);
    APEX_UNUSED(thread_data);
    timer_stack = new std::stack<std::shared_ptr<apex::task_wrapper> >();
    apex::register_thread("OpenMP Thread");
}

/* Event #2, thread end */
extern "C" void apex_thread_end(
    ompt_data_t *thread_data              /* data of thread                      */
) {
    APEX_UNUSED(thread_data);
    apex::exit_thread();
    if (timer_stack != nullptr) { 
        delete(timer_stack); 
        timer_stack = nullptr;
    }
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
    sprintf(regionIDstr, "OpenMP_PARALLEL_REGION: UNRESOLVED ADDR %p", codeptr_ra);
    apex_ompt_start(regionIDstr, parallel_data);
}

/* Event #4, parallel region end */
static void apex_parallel_region_end (
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *encountering_task_data,  /* data of encountering task           */
    ompt_invoker_t invoker,               /* invoker of master task              */
    const void *codeptr_ra                /* return address of runtime call      */
) {
    apex_ompt_stop("OpenMP_PARALLEL_REGION", parallel_data);
}

/* Event #5, task create */
extern "C" void apex_task_create (
    ompt_data_t *encountering_task_data,         /* data of parent task                 */
    const omp_frame_t *encountering_task_frame,  /* frame data for parent task          */
    ompt_data_t *new_task_data,                  /* data of created task                */
    int type,                                    /* type of created task                */
    int has_dependences,                         /* created task has dependences        */
    const void *codeptr_ra                       /* return address of runtime call      */
) {
    char regionIDstr[128] = {0}; 
    if (codeptr_ra != NULL) {
        sprintf(regionIDstr, "OpenMP_TASK: UNRESOLVED ADDR %p", codeptr_ra);
    } else {
        sprintf(regionIDstr, "OpenMP_TASK");
    }
    apex::sample_value(std::string(regionIDstr),1);
}
 
/* Event #6, task schedule */
extern "C" void apex_task_schedule(
    ompt_data_t *prior_task_data,         /* data of prior task                  */
    ompt_task_status_t prior_task_status, /* status of prior task                */
    ompt_data_t *next_task_data           /* data of next task                   */
    ) {
    if (prior_task_data != nullptr) {
        if (prior_task_status == ompt_task_complete) {
            apex_ompt_task_stop("OpenMP_TASK", prior_task_data);
        } else if (prior_task_status == ompt_task_yield) {
            apex_ompt_task_stop("OpenMP_TASK", prior_task_data);
        } else if (prior_task_status == ompt_task_cancel) {
            apex_ompt_task_stop("OpenMP_TASK", prior_task_data);
        } else if (prior_task_status == ompt_task_others) {
        }
    }
    apex_ompt_task_start("OpenMP_TASK", next_task_data);
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
        apex::sample_value(std::string("OpenMP_IMPLICIT_TASK"),1);
        //fprintf(stderr,"implicit task start, %u of %u : %lu\n", thread_num, team_size, task_data->value); fflush(stderr);
        apex_ompt_task_start("OpenMP_IMPLICIT_TASK", task_data);
    } else {
        //fprintf(stderr,"implicit task stop, %u of %u : %lu\n", thread_num, team_size, task_data->value); fflush(stderr);
        apex_ompt_task_stop("OpenMP_IMPLICIT_TASK", task_data);
    }
}

extern "C" void apex_target (
    ompt_target_type_t kind,
    ompt_scope_endpoint_t endpoint,
    uint64_t device_num,
    ompt_data_t *task_data,
    ompt_id_t target_id,
    const void *codeptr_ra
) {
}

extern "C" void apex_target_data_op (
    ompt_id_t target_id,
    ompt_id_t host_op_id,
    ompt_target_data_op_t optype,
    void *host_addr,
    void *device_addr,
    size_t bytes
) {
}

extern "C" void apex_target_submit (
    ompt_id_t target_id,
    ompt_id_t host_op_id
) {
}

extern "C" void apex_device_initialize (
    uint64_t device_num,
    const char *type,
    ompt_device_t *device,
    ompt_function_lookup_t lookup,
    const char *documentation
);

extern "C" void apex_device_finalize (
    uint64_t device_num
);

extern "C" void apex_device_load_t (
    uint64_t device_num,
    const char * filename,
    int64_t offset_in_file,
    void * vma_in_file,
    size_t bytes,
    void * host_addr,
    void * device_addr,
    uint64_t module_id
);

extern "C" void apex_device_unload (
    uint64_t device_num,
    uint64_t module_id
);

extern "C" void apex_control(uint64_t command, uint64_t modifier) {
  APEX_UNUSED(command);
  APEX_UNUSED(modifier);
}

/**********************************************************************/
/* End Mandatory Events */
/**********************************************************************/

/**********************************************************************/
/* Macros for common begin / end functionality. */
/**********************************************************************/

extern "C" void apex_ompt_work (
    ompt_work_type_t wstype,              /* type of work region                 */
    ompt_scope_endpoint_t endpoint,       /* endpoint of work region             */
    ompt_data_t *parallel_data,           /* data of parallel region             */
    ompt_data_t *task_data,               /* data of task                        */
    uint64_t count,                       /* quantity of work                    */
    const void *codeptr_ra                /* return address of runtime call      */
    ) {

    if (wstype == ompt_work_sections) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_SECTIONS", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_SECTIONS", parallel_data);
        }
    } else if (wstype == ompt_work_single_executor) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_SINGLE_EXECUTOR", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_SINGLE_EXECUTOR", parallel_data);
        }
    } else if (wstype == ompt_work_single_other) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_SINGLE_OTHER", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_SINGLE_OTHER", parallel_data);
        }
    } else if (wstype == ompt_work_workshare) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_WORKSHARE", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_WORKSHARE", parallel_data);
        }
    } else if (wstype == ompt_work_distribute) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_DISTRIBUTE", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_DISTRIBUTE", parallel_data);
        }
    } else if (wstype == ompt_work_taskloop) {
        if (endpoint == ompt_scope_begin) {
            apex_ompt_start("OpenMP_TASKLOOP", parallel_data);
        } else {
            apex_ompt_stop("OpenMP_TASKLOOP", parallel_data);
        }
    }
}

// This macro is for checking that the function registration worked.
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

    apex::init("OPENMP_PROGRAM",0,1);
    timer_stack = new std::stack<std::shared_ptr<apex::task_wrapper> >();
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
    // Event 5: task create
    apex_ompt_register(ompt_callback_task_create,
        (ompt_callback_t)&apex_task_create, "task_create");
    // Event 6: task schedule (start/stop)
    apex_ompt_register(ompt_callback_task_schedule,
        (ompt_callback_t)&apex_task_schedule, "task_schedule");
    // Event 7: implicit task (start/stop)
    apex_ompt_register(ompt_callback_implicit_task,
        (ompt_callback_t)&apex_implicit_task, "implicit_task");
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
    // Event 13: device finalize
    // Event 14: device load
    // Event 15: device unload

    /* optional events */

    if (!apex::apex_options::ompt_required_events_only()) {
        apex_ompt_register(ompt_callback_work,
            (ompt_callback_t)&apex_ompt_work, "work");
        if (apex::apex_options::ompt_high_overhead_events()) {
        }
    /* Event 16: sync region wait begin or end   */
    /* Event 17: mutex released                  */
    /* Event 18: report task dependences         */
    /* Event 19: report task dependence          */
    /* Event 20: task at work begin or end       */
    /* Event 21: task at master begin or end     */
    /* Event 22: target map                      */
    /* Event 23: sync region begin or end        */
    /* Event 24: lock init                       */
    /* Event 25: lock destroy                    */
    /* Event 26: mutex acquire                   */
    /* Event 27: mutex acquired                  */
    /* Event 28: nest lock                       */
    /* Event 29: after executing flush           */
    /* Event 30: cancel innermost binding region */
    /* Event 31: begin or end idle state         */

    }

    fprintf(stderr,"done.\n"); fflush(stderr);
    return 1;
}

void ompt_finalize(ompt_data_t* tool_data)
{
    printf("OpenMP runtime is shutting down...\n");
    if (timer_stack != nullptr) { 
        delete(timer_stack); 
        timer_stack = nullptr;
    }
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
