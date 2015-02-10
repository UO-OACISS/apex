#include "apex_global.h"
#include "mpi.h"
#include "stdlib.h" // malloc, etc.
#include <string.h> // memcpy, etc.
#include "math.h"
#include "stdio.h"

// my local value, global to this process
apex_profile value;

// the global reduced value
apex_profile reduced_value;

// the profiled function
apex_function_address profiled_action;

FILE *graph_output;

// global mpi variables
int rank, size;

// window location
apex_profile * inValues = NULL;
MPI_Win profile_window;
MPI_Group window_group;
MPI_Datatype profile_type;
size_t apex_profile_size;

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

void apex_sum(int count, apex_profile values[count]) {
  int i;
  memset(&reduced_value, 0, apex_profile_size);
  for (i = 0; i < count; ++i) { 
    reduced_value.calls += values[i].calls; 
    reduced_value.accumulated += values[i].accumulated; 
    reduced_value.sum_squares += values[i].sum_squares; 
    reduced_value.minimum = min(reduced_value.minimum,values[i].minimum); 
    reduced_value.maximum = max(reduced_value.maximum,values[i].maximum); 
  }
  return ;
}

// update our local value for the profile
int action_apex_get_value(void *args) {
  apex_profile * p = apex_get_profile_from_address(profiled_action);
  if (p != NULL) {
    value.calls = p->calls;
    value.accumulated = p->accumulated;
    value.sum_squares = p->sum_squares;
    value.minimum = p->minimum;
    value.maximum = p->maximum;
  }
  return 0;
}

int action_apex_reduce(void *unused) {
  int target_rank = 0;
  action_apex_get_value(NULL);

  if (rank != 0) {
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, profile_window);
    MPI_Put(&value, apex_profile_size, MPI_BYTE, 0, rank, apex_profile_size, MPI_BYTE, profile_window);
    MPI_Win_unlock(0, profile_window);
  } else {
    memcpy(&(inValues[0]), &value, apex_profile_size);
    apex_sum(size, inValues);
  }
  return 0;
}

int apex_periodic_policy_func(apex_context const context) {
  action_apex_reduce(NULL);
  if (rank != 0) return 1;
  double avg = 0.0;
  double stddev = 0.0;
  if (reduced_value.calls > 0.0) {
    avg = reduced_value.accumulated / reduced_value.calls;
    stddev = sqrt((reduced_value.sum_squares / reduced_value.calls) - (avg*avg));
  }
  printf("Function calls=%lu mean=%f +/- %f\r", (unsigned long)reduced_value.calls, avg, stddev);
  fflush(stdout);
  fprintf(graph_output,"%f %f\n",avg, stddev);
  fflush(graph_output);
  return 1;
}

void apex_global_setup(apex_function_address in_action) {
  profiled_action = in_action;
  apex_register_periodic_policy(1000000, apex_periodic_policy_func);
  apex_set_use_policy(true);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  apex_profile_size = sizeof(apex_profile);
  MPI_Comm_group(MPI_COMM_WORLD, &window_group);
  if (rank == 0) {
    graph_output = fopen("./profile_data.txt", "w");
    //inValues = (apex_profile*)(malloc(apex_profile_size * size));
	MPI_Alloc_mem(size * apex_profile_size, MPI_INFO_NULL, &inValues);
	memset(&(inValues[0]), 0, (apex_profile_size * size));
    MPI_Win_create(inValues, apex_profile_size * size, apex_profile_size, 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &profile_window);
  } else {
    MPI_Win_create(inValues, 0, apex_profile_size, 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &profile_window);
  }
}

void apex_global_teardown(void) {
  printf("\n");
  fflush(stdout);
  // This teardown process is causing crashes on some platforms. Disabled for now.
  return;
#if 0
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&profile_window); 
  if (rank == 0) {
    fclose(graph_output);
    MPI_Free_mem(inValues);
  }
#endif
}

