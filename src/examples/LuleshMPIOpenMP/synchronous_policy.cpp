#include <mpi.h>
#include <iostream>
#include <array>
#include <algorithm>
#include <atomic>
#if defined (_OPENMP)
#include "omp.h"
#else
#define omp_get_max_threads() 1
#endif
#include "apex_api.hpp"
#include "apex_global.h"

long __active_threads = omp_get_max_threads();
double accumulated_aggregate;
int __myrank = 0;
int __num_ranks = 1;
double previous_value = 0.0;
double current_value = 0.0;
apex_function_address function_address;

void apex_example_set_function_address(apex_function_address addr) {
  function_address = addr;
}

int apex_example_policy_func(apex_context const context) {
  APEX_UNUSED(context);
  // get value
  apex_profile * p = NULL;
  p = apex_get_profile(APEX_FUNCTION_ADDRESS, (void*)function_address);
  if (p != NULL) {
    previous_value = current_value;
    current_value = p->accumulated;
  }
  // wait for the application to warm up
  static int countdown = 10;
  if (countdown > 0) {
    countdown = countdown - 1;
    return APEX_NOERROR;
  }
  //countdown = 10;
  // get latest from last period
  double values[2];
  values[0] = (double)__active_threads;
  double mytimer = current_value - previous_value;
  values[1] = mytimer;
  double outvalues[2];
  // "reduce" the thread caps, timers
  PMPI_Allreduce(values, outvalues, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  outvalues[0] = outvalues[0] / __num_ranks;
  outvalues[1] = outvalues[1] / __num_ranks;
  // should we change our cap? - are we off by more than 1/2 of a thread?
  //double one_worker = (1.0/apex::apex_options::throttling_max_threads())*0.55;
  double one_worker = (1.0/__active_threads)*0.51;
  if (((mytimer/outvalues[1]) > ((__active_threads/outvalues[0]) + one_worker))
      && (__active_threads < apex::apex_options::throttling_max_threads())) {
      /*
    std::cout << __myrank << ": " << one_worker << " timer: " << (mytimer/outvalues[1])
	  << " thread: " << (__active_threads/outvalues[0]) << std::endl;
      */
    __active_threads++;
    /*
    std::cout << __myrank << ": New thread count: " << __active_threads << std::endl;
    */
  } else if (((mytimer/outvalues[1]) < ((__active_threads/outvalues[0]) - one_worker))
      && (__active_threads > apex::apex_options::throttling_min_threads())) {
    /*
    std::cout << __myrank << ": " << one_worker << " timer: " << (mytimer/outvalues[1])
	  << " thread: " << (__active_threads/outvalues[0]) << std::endl;
    */
    __active_threads--;
    /*
    std::cout << __myrank << ": New thread count: " << __active_threads << std::endl;
    */
  }
  return APEX_NOERROR;
}

long apex_example_get_active_threads(void) {
  return __active_threads;
}

void apex_example_set_rank_info(int me, int all) {
  __myrank = me;
  __num_ranks = all;
}
