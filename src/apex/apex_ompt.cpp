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

std::unordered_map<uint64_t, void*> task_addresses;
#define NUM_REGIONS 128
std::atomic<const void*> parallel_regions[NUM_REGIONS];
apex::shared_mutex_type _region_mutex;

APEX_NATIVE_TLS std::stack<apex::profiler*> *timer_stack;

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

void format_name_fast(const char * state, uint64_t parallel_id, char * result) {
	// get the parallel region address
    const void* ip = parallel_regions[(parallel_id%NUM_REGIONS)];
	// format it.
    sprintf(result, "%s: UNRESOLVED ADDR %p", state, (const void*)ip);
}

void format_task_name_fast(const char * state, uint64_t task_id, char * result) {
	// get the parallel region address
	std::unordered_map<uint64_t, void*>::const_iterator got;
	{
    	apex::read_lock_type l(_region_mutex);
	    got = task_addresses.find (task_id);
	}
	if ( got == task_addresses.end() ) { // not found.
		// format it.
    	sprintf(result, "%s", state);
	} else {
		// format it.
    	sprintf(result, "%s: UNRESOLVED ADDR %p", state, (void*)got->second);
	}
}

void apex_ompt_idle_start() {
  //fprintf(stderr,"start %s : %lu\n",state, parallel_id); fflush(stderr);
  const std::string regionIDstr("OpenMP_IDLE"); 
  apex::profiler* p = apex::start(regionIDstr);
  timer_stack->push(p);
}

void apex_ompt_task_start(const char * state, ompt_data_t * task_id) {
  //fprintf(stderr,"start %s : %lu\n",state, parallel_id); fflush(stderr);
  char regionIDstr[128] = {0}; 
  format_task_name_fast(state, task_id->value, regionIDstr);
  //apex::profiler* p = apex::start(std::string(regionIDstr));
  //timer_stack->push(p);
  apex::sample_value(std::string(regionIDstr),1);
}

void apex_ompt_pr_start(const char * state, ompt_data_t * parallel_id) {
  //fprintf(stderr,"start %s : %lu\n",state, parallel_id); fflush(stderr);
  char regionIDstr[128] = {0}; 
  format_name_fast(state, parallel_id->value, regionIDstr);
  apex::profiler* p = apex::start(std::string(regionIDstr));
  timer_stack->push(p);
}

void apex_ompt_start(const char * state, ompt_data_t * parallel_id) {
  //fprintf(stderr,"start %s : %lu\n",state, parallel_id); fflush(stderr);
  char regionIDstr[128] = {0}; 
  format_name_fast(state, parallel_id->value, regionIDstr);
  apex::profiler* p = apex::start(std::string(regionIDstr));
  timer_stack->push(p);
}

void apex_ompt_stop(const char * state, ompt_data_t * parallel_id) {
  //fprintf(stderr,"stop %s : %lu\n",state, parallel_id); fflush(stderr);
  APEX_UNUSED(state);
  APEX_UNUSED(parallel_id);
  if (timer_stack->empty()) { // uh-oh...
    apex::profiler * p = nullptr;
    apex::stop(p);
  } else {
    apex::profiler* p = timer_stack->top();
    apex::stop(p);
    timer_stack->pop();
  }
}

/*
 * Mandatory Events
 * 
 * The following events are supported by all OMPT implementations.
 */

static void apex_parallel_region_begin (
  ompt_data_t *parent_task_id,    /* id of parent task            */
  const omp_frame_t *parent_task_frame,  /* frame data of parent task    */
  ompt_data_t *parallel_id,   /* id of parallel region        */
  uint32_t requested_team_size,     /* Region team size             */
  ompt_invoker_t invoker,
  const void *parallel_function)          /* pointer to outlined function */
{
  APEX_UNUSED(parent_task_id);
  APEX_UNUSED(parent_task_frame);
  APEX_UNUSED(requested_team_size);
  //fprintf(stderr,"begin: %lu, %p, %lu, %u, %p\n", parent_task_id, parent_task_frame, parallel_id, requested_team_size, parallel_function); fflush(stderr);
  parallel_regions[(parallel_id->value%NUM_REGIONS)] = parallel_function;
  apex_ompt_pr_start("OpenMP_PARALLEL_REGION", parallel_id);
}

static void apex_parallel_region_end (
  ompt_data_t *parallel_id,   /* id of parallel region        */
  ompt_data_t *parent_task_id,
  ompt_invoker_t invoker,
  const void *codeptr_ra)    /* id of parent task            */
{
  APEX_UNUSED(parent_task_id);
  APEX_UNUSED(parallel_id);
  apex_ompt_stop("OpenMP_PARALLEL_REGION", parallel_id);
}

extern "C" void apex_task_begin (
  ompt_data_t *parent_task_id,    /* id of parent task            */
  omp_frame_t *parent_task_frame,  /* frame data for parent task   */
  ompt_data_t *new_task_id,      /* id of created task           */
  void *task_function)              /* pointer to outlined function */
{
  APEX_UNUSED(parent_task_id);
  APEX_UNUSED(parent_task_frame);
  {
    apex::write_lock_type l(_region_mutex);
    task_addresses[new_task_id->value] = task_function;
  }
  apex_ompt_task_start("OpenMP_TASK", new_task_id);
}
 
extern "C" void apex_task_end (
  ompt_data_t *task_id)      /* id of task           */
{
  apex_ompt_stop("OpenMP_TASK", task_id);
}

extern "C" void apex_thread_begin(apex_ompt_thread_type_t thread_type, ompt_data_t thread_id) {
  APEX_UNUSED(thread_type);
  APEX_UNUSED(thread_id);
  timer_stack = new std::stack<apex::profiler*>();
  apex::register_thread("OpenMP Thread");
}

void cleanup(void) {
    if (timer_stack != nullptr) { 
        delete(timer_stack); 
        timer_stack = nullptr;
    }
}

extern "C" void apex_thread_end(apex_ompt_thread_type_t thread_type, ompt_data_t thread_id) {
  APEX_UNUSED(thread_type);
  APEX_UNUSED(thread_id);
  apex::dump(false);
  apex::exit_thread();
  cleanup();
}

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
  timer_stack = new std::stack<apex::profiler*>();
  fprintf(stderr,"Registering OMPT events..."); fflush(stderr);
  apex_ompt_register(ompt_callback_parallel_begin, (ompt_callback_t)&apex_parallel_region_begin, "parallel_begin");
  apex_ompt_register(ompt_callback_parallel_end, (ompt_callback_t)&apex_parallel_region_end, "parallel_end");

  apex_ompt_register(ompt_callback_task_create, (ompt_callback_t)&apex_task_begin, "task_begin");
  //apex_ompt_register(ompt_callback_task_end, (ompt_callback_t)&apex_task_end, "task_end");
  apex_ompt_register(ompt_callback_thread_begin, (ompt_callback_t)&apex_thread_begin, "thread_begin");
  apex_ompt_register(ompt_callback_thread_end, (ompt_callback_t)&apex_thread_end, "thread_end");
  apex_ompt_register(ompt_callback_control_tool, (ompt_callback_t)&apex_control, "event_control");

  //if (!apex::apex_options::ompt_required_events_only()) {
    apex_ompt_register(ompt_callback_work, (ompt_callback_t)&apex_ompt_work, "work");
    //if (apex::apex_options::ompt_high_overhead_events()) {
    //}
  //}
#if 0
  if (!apex::apex_options::ompt_required_events_only()) {
    //CHECK(ompt_event_wait_lock, apex_wait_lock, "wait_lock");
    //CHECK(ompt_event_wait_nest_lock, apex_wait_nest_lock, "wait_nest_lock");
    CHECK(ompt_event_wait_critical, apex_wait_critical, "wait_critical");
    CHECK(ompt_event_wait_atomic, apex_wait_atomic, "wait_atomic");
    CHECK(ompt_event_wait_ordered, apex_wait_ordered, "wait_ordered");

    //CHECK(ompt_event_acquired_lock, apex_acquired_lock, "acquired_lock");
    //CHECK(ompt_event_acquired_nest_lock, apex_acquired_nest_lock, "acquired_nest_lock");
    CHECK(ompt_event_acquired_critical, apex_acquired_critical, "acquired_critical");
    CHECK(ompt_event_acquired_atomic, apex_acquired_atomic, "acquired_atomic");
    CHECK(ompt_event_acquired_ordered, apex_acquired_ordered, "acquired_ordered");

    //CHECK(ompt_event_release_lock, apex_release_lock, "release_lock");
    //CHECK(ompt_event_release_nest_lock, apex_release_nest_lock, "release_nest_lock");
    CHECK(ompt_event_release_critical, apex_release_critical, "release_critical");
    CHECK(ompt_event_release_atomic, apex_release_atomic, "release_atomic");
    CHECK(ompt_event_release_ordered, apex_release_ordered, "release_ordered");

    CHECK(ompt_event_barrier_begin, apex_barrier_begin, "barrier_begin");
    CHECK(ompt_event_barrier_end, apex_barrier_end, "barrier_end");
    CHECK(ompt_event_master_begin, apex_master_begin, "master_begin");
    CHECK(ompt_event_master_end, apex_master_end, "master_end");
    if (apex::apex_options::ompt_high_overhead_events()) {
      CHECK(ompt_event_sections_begin, apex_sections_begin, "sections_begin");
      CHECK(ompt_event_sections_end, apex_sections_end, "sections_end");
      CHECK(ompt_event_taskwait_begin, apex_taskwait_begin, "taskwait_begin");
      CHECK(ompt_event_taskwait_end, apex_taskwait_end, "taskwait_end");
      CHECK(ompt_event_taskgroup_begin, apex_taskgroup_begin, "taskgroup_begin");
      CHECK(ompt_event_taskgroup_end, apex_taskgroup_end, "taskgroup_end");
      CHECK(ompt_event_workshare_begin, apex_workshare_begin, "workshare_begin");
      CHECK(ompt_event_workshare_end, apex_workshare_end, "workshare_end");

    /* These are high overhead events! */
      CHECK(ompt_event_implicit_task_begin, apex_implicit_task_begin, "task_begin");
      CHECK(ompt_event_implicit_task_end, apex_implicit_task_end, "task_end");
      CHECK(ompt_event_idle_begin, apex_idle_begin, "idle_begin");
      CHECK(ompt_event_idle_end, apex_idle_end, "idle_end");
	}
  }
#endif
  fprintf(stderr,"done.\n"); fflush(stderr);
  return 1;
}

void ompt_finalize(ompt_data_t* tool_data)
{
    printf("OpenMP runtime is shutting down...\n");
    cleanup();
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
