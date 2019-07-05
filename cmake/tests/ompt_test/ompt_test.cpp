#define _BSD_SOURCE

#include "stdio.h"
#include <omp.h>
#include <ompt.h>
#include <iostream>

#define cb_t(name) (ompt_callback_t)&name

/* Function pointers.  These are all queried from the runtime during
 * ompt_initialize() */
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

static void
on_ompt_callback_parallel_begin(
  ompt_data_t *parent_task_data,
  const ompt_frame_t *parent_task_frame,
  ompt_data_t* parallel_data,
  uint32_t requested_team_size,
  int flags,
  const void *codeptr_ra)
{
  printf("parallel_begin\n");
}

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  int flags,
  const void *codeptr_ra)
{
  printf("parallel_end\n");
}

/* Register callbacks. This function is invoked only from the ompt_start_tool routine.
 * Callbacks that only have "ompt_set_always" are the required events that we HAVE to support */
inline static void register_callback(ompt_callbacks_t name, ompt_callback_t cb) {
  int ret = ompt_set_callback(name, cb);

  switch(ret) { 
    case ompt_set_never:
      fprintf(stderr, "TAU: WARNING: Callback for event %d could not be registered\n", name); 
      break; 
    case ompt_set_sometimes: 
      printf("TAU: Callback for event %d registered with return value %s\n", name, "ompt_set_sometimes");
      break;
    case ompt_set_sometimes_paired:
      printf("TAU: Callback for event %d registered with return value %s\n", name, "ompt_set_sometimes_paired");
      break;
    case ompt_set_always:
      printf("TAU: Callback for event %d registered with return value %s\n", name, "ompt_set_always");
      break;
  }
}


/* Register callbacks for all events that we are interested in / have to support */
extern "C" int ompt_initialize(
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t* tool_data)
{
  int ret;

/* Gather the required function pointers using the lookup tool */
  printf("Registering OMPT events...\n"); fflush(stdout);
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

/* Required events */
  register_callback(ompt_callback_parallel_begin, cb_t(on_ompt_callback_parallel_begin));
  register_callback(ompt_callback_parallel_end, cb_t(on_ompt_callback_parallel_end));
  //register_callback(ompt_callback_task_create, cb_t(on_ompt_callback_task_create));
  //register_callback(ompt_callback_task_schedule, cb_t(on_ompt_callback_task_schedule));
  //register_callback(ompt_callback_implicit_task, cb_t(on_ompt_callback_implicit_task)); //Sometimes high-overhead, but unfortunately we cannot avoid this as it is a required event 
  //register_callback(ompt_callback_thread_begin, cb_t(on_ompt_callback_thread_begin));
  //register_callback(ompt_callback_thread_end, cb_t(on_ompt_callback_thread_end));

  return 1; //success
}

extern "C" void ompt_finalize(ompt_data_t* tool_data) {
}

extern "C" ompt_start_tool_result_t * ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t result;
  result.initialize = &ompt_initialize;
  result.finalize = &ompt_finalize;
  result.tool_data.value = 0L;
  result.tool_data.ptr = NULL;
  std::cout << omp_version << std::endl;
  return &result;
}

int main (int argc, char * argv[]) {
#pragma omp parallel
    {
        printf ("Hi from thread %d\n", omp_get_thread_num());
    }
}
