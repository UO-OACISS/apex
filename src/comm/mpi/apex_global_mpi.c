#include "apex_global.h"
#include "mpi.h"
#include "stdlib.h" // malloc, etc.
#include <string.h> // memcpy, etc.
#include "math.h"
#include "stdio.h"
#include <float.h> // DBL_MAX
#ifndef __clang__ // no OpenMP support.
#include "omp.h"
#endif

#define APEX_LOCALITY 0 // process that will do the reduction
// my local value, global to this process
apex_profile value;

// the global reduced value
apex_profile reduced_value;
int min_rank = 0;
int max_rank = 0;
int global_max_threads = 1;
int* thread_caps = NULL;
double* previous_values = NULL;
double* current_values = NULL;

// the profiled function
apex_function_address profiled_action = 0L;
char* profiled_action_name;
apex_profiler_type profiler_type;
bool good_values = false;

FILE *graph_output;

// global mpi variables
int rank, num_ranks;

// window location
apex_profile * inValues = NULL;
MPI_Win profile_window;
MPI_Group window_group;
MPI_Datatype profile_type;
size_t apex_profile_size;

static bool _finalized = false;
bool apex_set_new_thread_caps(int count, apex_profile * values) {
  static int countdown = 5; // wait for the application to warm up
  if (countdown > 0) {
    countdown = countdown - 1;
	printf("Waiting...\n");
    return false;
  }
  // divide max by global_max_threads - or the current thread cap for the max rank
  double period_max = (reduced_value.maximum - previous_values[max_rank]);
  //double period_max = reduced_value.maximum;
  // set each node thread cap to a relative multiple of that
  int i;
  for (i = 0; i < count; ++i) { 
    int old_cap = thread_caps[i];
    // how much accumulated in the last period?
    double last_period = values[i].accumulated - previous_values[i];
    //double last_period = values[i].accumulated;
    double ratio = last_period / period_max;
    // if this is the max rank, increase the thread count, limited to max.
    if (i == max_rank) {
      //thread_caps[i] = fmin(global_max_threads, (thread_caps[i]+1));
      thread_caps[i] = global_max_threads;
    } else {
      // if I did significantly less work than max, reduce my thread count
      int new_cap = fmin(global_max_threads, ceil(ratio * global_max_threads));
      if (new_cap != thread_caps[i]) {
        thread_caps[i] = fmax(apex_get_throttling_min_threads(), new_cap);
      }
    }
    printf("Locality %d: %f (%f) of new work from %d threads, new cap: %d    \n", i, last_period, ratio, old_cap, thread_caps[i]);
  }
  return true;
}


void apex_sum(int count, apex_profile * values) {
  // swap the current and previous values, for relative differences
  double* tmp = current_values;
  current_values = previous_values;
  previous_values = tmp;
  reduced_value.calls = values[0].calls; 
  reduced_value.accumulated = values[0].accumulated; 
  reduced_value.sum_squares = values[0].sum_squares; 
  //reduced_value.minimum = values[0].minimum; 
  //reduced_value.maximum = values[0].maximum; 
  reduced_value.minimum = values[0].accumulated;
  reduced_value.maximum = values[0].accumulated;
  current_values[0] = values[0].accumulated;
  min_rank = max_rank = 0;
  int local_good_values = (values[0].calls > 0.0 && values[0].accumulated > 0.0) ? 1 : 0;
  int i;
  for (i = 1; i < count; ++i) { 
    reduced_value.calls += values[i].calls; 
    reduced_value.accumulated += values[i].accumulated; 
    reduced_value.sum_squares += values[i].sum_squares; 
    //reduced_value.minimum = fmin(reduced_value.minimum,values[i].minimum); 
      double accum = values[i].accumulated;
      if (accum < reduced_value.minimum) {
        reduced_value.minimum = accum; 
        min_rank = i;
      }
      //reduced_value.maximum = fmax(reduced_value.maximum,values[i].maximum); 
      if (accum > reduced_value.maximum) {
        reduced_value.maximum = accum; 
        max_rank = i;
      }
      current_values[i] = values[i].accumulated;
      local_good_values = local_good_values + ((values[i].calls > 0) ? 1 : 0);
  }
  printf("good values: %d of %d\n", local_good_values, count);
  if (local_good_values == count) good_values = true;
  return ;
}

// update our local value for the profile
int action_apex_get_value(void *args) {
  apex_profile * p = NULL;
  if (profiler_type == APEX_FUNCTION_ADDRESS) {
    p = apex_get_profile(APEX_FUNCTION_ADDRESS, (void*)profiled_action);
  } else {
    p = apex_get_profile(APEX_NAME_STRING, profiled_action_name);
  }
  if (p != NULL) {
    value.calls = p->calls;
    value.accumulated = p->accumulated;
    value.sum_squares = p->sum_squares;
    value.minimum = p->minimum;
    value.maximum = p->maximum;
  }
  return APEX_NOERROR;
}

int apex_set_local_cap(void *args, size_t size) {
  int local_cap = *(int*)args;
  printf("At rank %d received value %d\n", rank, local_cap);
  apex_set_thread_cap(local_cap);
  return APEX_NOERROR;
}

int action_apex_reduce(void *unused) {
  //int target_rank = 0;
  action_apex_get_value(NULL);

  if (rank != 0) {
    //printf("\nRank %d locking window...\n", rank);
    MPI_Win_lock(MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, profile_window);
    //MPI_Win_lock_all(0, profile_window);
    //MPI_Win_fence(0, profile_window);
    //printf("\nRank %d putting values...\n", rank);
    MPI_Put(&value, apex_profile_size, MPI_BYTE, 0, rank, apex_profile_size, MPI_BYTE, profile_window);
    //printf("\nRank %d unlocking window...\n", rank);
    MPI_Win_unlock(0, profile_window);
    //MPI_Win_unlock_all(profile_window);
    //printf("\nRank %d done.\n", rank);
  } else {
    memcpy(&(inValues[0]), &value, apex_profile_size);
    apex_sum(num_ranks, inValues);
    bool ready = apex_set_new_thread_caps(num_ranks, inValues);

    if (ready && good_values) {
      // set new caps for everyone
      int i;
      for (i = 0; i < num_ranks; ++i) {
      }
    }
  }
  return APEX_NOERROR;
}

int apex_periodic_policy_func(apex_context const context) {
  if (_finalized) return APEX_NOERROR;
  action_apex_reduce(NULL);
  if (rank != 0) return APEX_NOERROR;
  double avg = 0.0;
  double stddev = 0.0;
  if (reduced_value.calls > 0.0) {
    avg = reduced_value.accumulated / reduced_value.calls;
    stddev = sqrt((reduced_value.sum_squares / reduced_value.calls) - (avg*avg));
  }
  printf("Function calls=%lu min=%f mean=%f max=%f +/- %f\r", (unsigned long)reduced_value.calls, reduced_value.minimum, avg, reduced_value.maximum, stddev);
  fflush(stdout);
  fprintf(graph_output,"%f %f\n",avg, stddev);
  fflush(graph_output);
  return APEX_NOERROR;
}

void apex_global_setup(apex_profiler_type type, void* in_action) {
  profiler_type = type;
  if (type == APEX_FUNCTION_ADDRESS) {
    profiled_action = (apex_function_address)in_action;
  } else {
    char* tmp = (char *)in_action;
    profiled_action_name = strdup(tmp);
  }
    
  apex_register_periodic_policy(1000000, apex_periodic_policy_func);
  apex_set_use_policy(true);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  apex_profile_size = sizeof(apex_profile);
  MPI_Comm_group(MPI_COMM_WORLD, &window_group);
  if (rank == 0) {
    graph_output = fopen("./profile_data.txt", "w");
    fprintf(graph_output,"\"calls\"\t\"mean\"\t\"stddev\"\t\"min\"\t\"min_rank\"\t\"max\"\t\"max_rank\"\n");
    fflush(graph_output);
    //inValues = (apex_profile*)(malloc(apex_profile_size * size));
    MPI_Alloc_mem(num_ranks * apex_profile_size, MPI_INFO_NULL, &inValues);
    memset(&(inValues[0]), 0, (apex_profile_size * num_ranks));
    MPI_Win_create(inValues, apex_profile_size * num_ranks, apex_profile_size, 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &profile_window);
    // get the max number of threads. All throttling will be done relative to this.
#pragma omp parallel
    {
#ifndef __clang__
    global_max_threads = omp_get_max_threads();
#endif
    printf("Got %d threads!\n", global_max_threads);
    }
    thread_caps = calloc(num_ranks, sizeof(int));
    int i;
    for (i = 0; i < num_ranks; ++i) { 
      thread_caps[i] = global_max_threads;
    }
    previous_values = calloc(num_ranks, sizeof(double));
    current_values = calloc(num_ranks, sizeof(double));
  } else {
    MPI_Win_create(inValues, 0, apex_profile_size, 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &profile_window);
  }
}

void apex_global_teardown(void) {
  _finalized = true;
  if (rank == 0) { printf("\n"); fflush(stdout); }
  // This teardown process is causing crashes on some platforms. Disabled for now.
#if 1
  MPI_Barrier(MPI_COMM_WORLD);
  /* Added a call to MPI_WIN_FENCE, per MPI-2.1 11.2.1 */
  /* Removed a call to MPI_WIN_FENCE, because it crashes. */
  //MPI_Win_fence(0, profile_window);
  if (profile_window != NULL) {
    MPI_Win_free(&profile_window); 
  }
  if (rank == 0) {
    fclose(graph_output);
    MPI_Free_mem(inValues);
  }
#endif
  return;
}

