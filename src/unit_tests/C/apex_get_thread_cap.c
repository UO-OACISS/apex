#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "apex.h"

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

#define NUM_THREADS 8
#define ITERATIONS 250
#define SLEEPY_TIME 10000 // 10,000

int total_iterations = NUM_THREADS * ITERATIONS;
bool test_passed = false;
int original_cap = NUM_THREADS;

int foo (int i) {
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &foo);
  int j = i*i;
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  int cap = MIN(NUM_THREADS,apex_get_thread_cap());
    if (cap != original_cap) {
        test_passed = true;
  }
  tim.tv_nsec = (unsigned long)(SLEEPY_TIME * randval * (cap * cap));
  nanosleep(&tim , &tim2);
  apex_stop(p);
  return j;
}

typedef void*(*start_routine_t)(void*);

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  int *myid = (int*)tmp;
  apex_register_thread("threadTest thread");
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  printf("The scheduler ID of this thread: %d\n", *myid);
  while (total_iterations > 0) {
      if (*myid >= apex_get_thread_cap()) {
        //printf("Thread %d sleeping for a bit.\n", *myid);
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 100000000; // 1/10 second
        // sleep a bit
        nanosleep(&tim , &tim2);
      } else {
        foo(total_iterations);
        __sync_fetch_and_sub(&(total_iterations),1);
        if (total_iterations % 1000 == 0) {
            printf("%d iterations left, cap is %d\n", total_iterations, apex_get_thread_cap());
        }
      }
  }
  printf("Thread done: %d. Current Cap: %d.\n", *myid, apex_get_thread_cap());
  apex_stop(p);
  apex_exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  apex_init_args(argc, argv, "apex_get_thread_cap unit test");
  apex_set_node_id(0);

  apex_setup_timer_throttling(APEX_FUNCTION_ADDRESS, &foo, APEX_MINIMIZE_ACCUMULATED,
          APEX_DISCRETE_HILL_CLIMBING, 1000000);

  original_cap = apex_get_thread_cap();

  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &main);
  //printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  int ids[NUM_THREADS];
  for (i = 0 ; i < NUM_THREADS ; i++) {
    ids[i] = i;
    pthread_create(&(thread[i]), NULL, someThread, &(ids[i]));
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop(p);
  int final_cap = apex_get_thread_cap();
  if (test_passed) {
    printf("Test passed. Final cap: %d\n", final_cap);
  }
  apex_shutdown_throttling();
  apex_finalize();
  return(0);
}

