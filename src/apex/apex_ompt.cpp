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

//#include "global_constructor_destructor.h"

typedef struct status_flags {
    char idle; // 4 bytes
    char busy; // 4 bytes
    char parallel; // 4 bytes
    char ordered_region_wait; // 4 bytes
    char ordered_region; // 4 bytes
    char task_exec; // 4 bytes
    char looping; // 4 bytes
    char acquired; // 4 bytes
    char waiting; // 4 bytes
    unsigned long regionid; // 8 bytes
    unsigned long taskid; // 8 bytes
    int *signal_message; // preallocated message for signal handling, 8 bytes
    int *region_message; // preallocated message for region handling, 8 bytes
    int *task_message; // preallocated message for task handling, 8 bytes
} status_flags_t;

#define OMPT_WAIT_ACQ_CRITICAL  0x01
#define OMPT_WAIT_ACQ_ORDERED   0x02
#define OMPT_WAIT_ACQ_ATOMIC    0x04
#define OMPT_WAIT_ACQ_LOCK      0x08
#define OMPT_WAIT_ACQ_NEST_LOCK 0x10

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
APEX_NATIVE_TLS status_flags_t *status;

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
  status = new status_flags();
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
  apex::exit_thread();
  delete(status);
  cleanup();
}

extern "C" void apex_control(uint64_t command, uint64_t modifier) {
  APEX_UNUSED(command);
  APEX_UNUSED(modifier);
}


/**********************************************************************/
/* End Mandatory Events */
/**********************************************************************/

#if 0

#define APEX_OMPT_WAIT_ACQUIRE_RELEASE(WAIT_FUNC,ACQUIRED_FUNC,RELEASE_FUNC,WAIT_NAME,REGION_NAME,CAUSE) \
extern "C" void WAIT_FUNC (omp_wait_id_t waitid) { \
  APEX_UNUSED(waitid); \
  if (status->waiting>0) { \
    apex_ompt_stop(WAIT_NAME,0); \
  } \
  apex_ompt_start(WAIT_NAME,0); \
  status->waiting = CAUSE; \
} \
 \
extern "C" void ACQUIRED_FUNC (omp_wait_id_t waitid) { \
  APEX_UNUSED(waitid); \
  if (status->waiting>0) { \
    apex_ompt_stop(WAIT_NAME,0); \
  } \
  status->waiting = 0; \
  apex_ompt_start(REGION_NAME,0); \
  status->acquired += CAUSE; \
} \
 \
extern "C" void RELEASE_FUNC (omp_wait_id_t waitid) { \
  APEX_UNUSED(waitid); \
  if (status->acquired>0) { \
    apex_ompt_stop(REGION_NAME,0); \
    status->acquired -= CAUSE; \
  } \
} \

APEX_OMPT_WAIT_ACQUIRE_RELEASE(apex_wait_atomic,apex_acquired_atomic,apex_release_atomic,"OpenMP_ATOMIC_REGION_WAIT","OpenMP_ATOMIC_REGION",OMPT_WAIT_ACQ_ATOMIC)
APEX_OMPT_WAIT_ACQUIRE_RELEASE(apex_wait_ordered,apex_acquired_ordered,apex_release_ordered,"OpenMP_ORDERED_REGION_WAIT","OpenMP_ORDERED_REGION",OMPT_WAIT_ACQ_ORDERED)
APEX_OMPT_WAIT_ACQUIRE_RELEASE(apex_wait_critical,apex_acquired_critical,apex_release_critical,"OpenMP_CRITICAL_REGION_WAIT","OpenMP_CRITICAL_REGION",OMPT_WAIT_ACQ_CRITICAL)
//APEX_OMPT_WAIT_ACQUIRE_RELEASE(apex_wait_lock,apex_acquired_lock,apex_release_lock,"OpenMP_LOCK_WAIT","OpenMP_LOCK",OMPT_WAIT_ACQ_LOCK)
//APEX_OMPT_WAIT_ACQUIRE_RELEASE(apex_wait_nest_lock,apex_acquired_nest_lock,apex_release_nest_lock,"OpenMP_LOCK_WAIT","OpenMP_LOCK",OMPT_WAIT_ACQ_NEST_LOCK)

#undef APEX_OMPT_WAIT_ACQUIRE_RELEASE

/**********************************************************************/
/* Macros for common begin / end functionality. */
/**********************************************************************/

#define APEX_OMPT_SIMPLE_BEGIN_AND_END(BEGIN_FUNCTION,END_FUNCTION,NAME) \
extern "C" void BEGIN_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  apex_ompt_start(NAME, parallel_id); \
} \
\
extern "C" void END_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  apex_ompt_stop(NAME, parallel_id); \
}

#define APEX_OMPT_TASK_BEGIN_AND_END(BEGIN_FUNCTION,END_FUNCTION,NAME) \
extern "C" void BEGIN_FUNCTION (ompt_data_t task_id) { \
  status->taskid = task_id; \
  apex_ompt_start(NAME, task_id); \
} \
\
extern "C" void END_FUNCTION (ompt_data_t task_id) { \
  status->taskid = task_id; \
  apex_ompt_stop(NAME, task_id); \
}

#define APEX_OMPT_LOOP_BEGIN_AND_END(BEGIN_FUNCTION,END_FUNCTION,NAME) \
extern "C" void BEGIN_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  apex_ompt_start(NAME, parallel_id); \
  status->looping=1; \
} \
\
extern "C" void END_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  if (status->looping==1) { \
  apex_ompt_stop(NAME, parallel_id); } \
  status->looping=0; \
}

#define APEX_OMPT_WORKSHARE_BEGIN_AND_END(BEGIN_FUNCTION,END_FUNCTION,NAME) \
extern "C" void BEGIN_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id, void *parallel_function) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  /*{ \
    apex::write_lock_type l(_region_mutex); \
    task_addresses[task_id->value] = parallel_function; \
  }*/ \
  apex_ompt_start(NAME, parallel_id); \
} \
\
extern "C" void END_FUNCTION (ompt_data_t parallel_id, ompt_data_t task_id) { \
  status->regionid = parallel_id; \
  status->taskid = task_id; \
  apex_ompt_stop(NAME, parallel_id); \
}

APEX_OMPT_TASK_BEGIN_AND_END(apex_initial_task_begin,apex_initial_task_end,"OpenMP_INITIAL_TASK")
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_barrier_begin,apex_barrier_end,"OpenMP_BARRIER")
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_implicit_task_begin,apex_implicit_task_end,"OpenMP_IMPLICIT_TASK")
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_wait_barrier_begin,apex_wait_barrier_end,"OpenMP_WAIT_BARRIER")
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_master_begin,apex_master_end,"OpenMP_MASTER_REGION")
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_single_others_begin,apex_single_others_end,"OpenMP_SINGLE_OTHERS") 
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_taskwait_begin,apex_taskwait_end,"OpenMP_TASKWAIT") 
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_wait_taskwait_begin,apex_wait_taskwait_end,"OpenMP_WAIT_TASKWAIT") 
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_taskgroup_begin,apex_taskgroup_end,"OpenMP_TASKGROUP") 
APEX_OMPT_SIMPLE_BEGIN_AND_END(apex_wait_taskgroup_begin,apex_wait_taskgroup_end,"OpenMP_WAIT_TASKGROUP") 
APEX_OMPT_WORKSHARE_BEGIN_AND_END(apex_loop_begin,apex_loop_end,"OpenMP_LOOP")
APEX_OMPT_WORKSHARE_BEGIN_AND_END(apex_single_in_block_begin,apex_single_in_block_end,"OpenMP_SINGLE_IN_BLOCK") 
APEX_OMPT_WORKSHARE_BEGIN_AND_END(apex_workshare_begin,apex_workshare_end,"OpenMP_WORKSHARE")
APEX_OMPT_WORKSHARE_BEGIN_AND_END(apex_sections_begin,apex_sections_end,"OpenMP_SECTIONS") 

#undef APEX_OMPT_SIMPLE_BEGIN_AND_END

/**********************************************************************/
/* Specialized begin / end functionality. */
/**********************************************************************/

/* Thread end idle */
extern "C" void apex_idle_end(ompt_data_t thread_id) {
  APEX_UNUSED(thread_id);
  apex_ompt_stop("OpenMP_IDLE", 0);
  //if (status->parallel==0) {
    //apex_ompt_start("OpenMP_PARALLEL_REGION", 0);
    //status->busy = 1;
  //}
  status->idle = 0;
}

/* Thread begin idle */
extern "C" void apex_idle_begin(ompt_data_t thread_id) {
  APEX_UNUSED(thread_id);
  if (status->parallel==0) {
    if (status->idle == 1 && status->busy == 0) {
        return;
    }
    if (status->busy == 1) {
        //apex_ompt_stop("OpenMP_PARALLEL_REGION", 0);
        status->busy = 0;
    }
  }
  status->idle = 1;
  apex_ompt_idle_start();
}

#endif // if 0


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
  status = new status_flags();
  fprintf(stderr,"Registering OMPT events..."); fflush(stderr);
  apex_ompt_register(ompt_callback_parallel_begin, (ompt_callback_t)&apex_parallel_region_begin, "parallel_begin");
  apex_ompt_register(ompt_callback_parallel_end, (ompt_callback_t)&apex_parallel_region_end, "parallel_end");

  apex_ompt_register(ompt_callback_task_create, (ompt_callback_t)&apex_task_begin, "task_begin");
  //apex_ompt_register(ompt_callback_task_end, (ompt_callback_t)&apex_task_end, "task_end");
  apex_ompt_register(ompt_callback_thread_begin, (ompt_callback_t)&apex_thread_begin, "thread_begin");
  apex_ompt_register(ompt_callback_thread_end, (ompt_callback_t)&apex_thread_end, "thread_end");
  apex_ompt_register(ompt_callback_control_tool, (ompt_callback_t)&apex_control, "event_control");

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
    CHECK(ompt_event_loop_begin, apex_loop_begin, "loop_begin");
    CHECK(ompt_event_loop_end, apex_loop_end, "loop_end");
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
    //delete(status);
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
