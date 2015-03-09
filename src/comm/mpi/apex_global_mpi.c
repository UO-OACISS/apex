#include "apex_global.h"
#include "mpi.h"
#include "stdlib.h" // malloc, etc.
#include <string.h> // memcpy, etc.
#include "math.h"
#include "stdio.h"
#include <float.h> // DBL_MAX

// my local value, global to this process
apex_profile* values;

// the global reduced value
apex_profile* reduced_values;

// the profiled function
apex_function_address* profiled_actions;
int num_profiles;

FILE *graph_output;

// global mpi variables
int rank, size;

// window location
apex_profile * inValues = NULL;
MPI_Win profile_window;
MPI_Group window_group;
MPI_Datatype profile_type;
size_t apex_profile_size;

static bool _finalized = false;

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

// values is a list of lists of profiles. That is, there is a list with "count" 
// lists, and each sub-list has "num_profiles" profiles in it.  "index" is the
// master index through the values array which is represented as 1-dimensional.
void apex_sum(int count, apex_profile values[]) {
    int i, n, index = 0;
    // initialize the root aggregation of profiles
    memset(reduced_values, 0, (apex_profile_size * num_profiles));
    for (n = 0; n < num_profiles ; n++) {
        reduced_values[n].minimum = DBL_MAX;
    }
    // iterate over the number of processes...
    for (i = 0; i < count; ++i) { 
        // for each process, iterate over its list of profiles.
        for (n = 0; n < num_profiles ; n++) {
            reduced_values[n].calls += values[index].calls; 
            reduced_values[n].accumulated += values[index].accumulated; 
            reduced_values[n].sum_squares += values[index].sum_squares; 
            if (values[index].minimum > 0.0) {
                reduced_values[n].minimum = min(reduced_values[n].minimum,values[index].minimum); 
            }
            reduced_values[n].maximum = max(reduced_values[n].maximum,values[index].maximum); 
            index++;
        }
    }
    // fix any min values that weren't updated.
    for (n = 0; n < num_profiles ; n++) {
        if (reduced_values[n].minimum == DBL_MAX) reduced_values[n].minimum = 0.0;
    }
    return ;
}

// update our local value for the profile
int action_apex_get_value(void *args) {
    int n;
    for (n = 0; n < num_profiles ; n++) {
        apex_profile * p = apex_get_profile_from_address(profiled_actions[n]);
        if (p != NULL) {
            values[n].calls = p->calls;
            values[n].accumulated = p->accumulated;
            values[n].sum_squares = p->sum_squares;
            values[n].minimum = p->minimum;
            values[n].maximum = p->maximum;
        }
    }
    return APEX_NOERROR;
}

int action_apex_reduce(void *unused) {
    int target_rank = 0;
    action_apex_get_value(NULL);

    if (rank != 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, profile_window);
        MPI_Put(values, (apex_profile_size * num_profiles), MPI_BYTE, 0, rank, 
                (apex_profile_size * num_profiles), MPI_BYTE, profile_window);
        MPI_Win_unlock(0, profile_window);
    } else {
        memcpy(inValues, values, (apex_profile_size * num_profiles));
        apex_sum(size, inValues);
    }
    return APEX_NOERROR;
}

int apex_periodic_policy_func(apex_context const context) {
    if (_finalized) return APEX_NOERROR;
    action_apex_reduce(NULL);
    if (rank != 0) return APEX_NOERROR;
    int n;
    for (n = 0; n < num_profiles ; n++) {
        double avg = 0.0;
        double stddev = 0.0;
        if (reduced_values[n].calls > 0.0) {
            avg = reduced_values[n].accumulated / reduced_values[n].calls;
            stddev = sqrt((reduced_values[n].sum_squares / reduced_values[n].calls) - (avg*avg));
        }
        printf("Function calls=%lu min=%f mean=%f max=%f +/- %f\n", (unsigned long)reduced_values[n].calls, reduced_values[n].minimum, avg, reduced_values[n].maximum, stddev);
        fflush(stdout);
        fprintf(graph_output,"%f %f ", avg, stddev);
        fflush(graph_output);
    }
    fprintf(graph_output,"\n");
    fflush(graph_output);
    return APEX_NOERROR;
}

void apex_global_setup(apex_function_address in_action) {
    num_profiles = 1;
    profiled_actions = calloc(num_profiles, sizeof(apex_function_address));
    apex_profile_size = sizeof(apex_profile);
    values = calloc(num_profiles, apex_profile_size);
    reduced_values = calloc(num_profiles, apex_profile_size);
    profiled_actions[0] = in_action;
    apex_register_periodic_policy(1000000, apex_periodic_policy_func);
    apex_set_use_policy(true);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_group(MPI_COMM_WORLD, &window_group);
    if (rank == 0) {
        graph_output = fopen("./profile_data.txt", "w");
        int bytes_to_alloc = size * num_profiles * apex_profile_size;
	    MPI_Alloc_mem(bytes_to_alloc, MPI_INFO_NULL, &inValues);
	    memset(&(inValues[0]), 0, bytes_to_alloc);
        MPI_Win_create(inValues, bytes_to_alloc, num_profiles * apex_profile_size, 
                    MPI_INFO_NULL, MPI_COMM_WORLD, &profile_window);
    } else {
        MPI_Win_create(inValues, 0, num_profiles * apex_profile_size, 
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
  MPI_Win_fence(0, profile_window);
  MPI_Win_free(&profile_window); 
  if (rank == 0) {
    fclose(graph_output);
    MPI_Free_mem(inValues);
  }
#endif
  return;
}

