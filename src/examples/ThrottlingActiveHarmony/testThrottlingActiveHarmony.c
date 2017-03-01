#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "apex.h"

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

#define ITERATIONS 1000
#define SLEEPY_TIME 10000 // 10,000

int num_threads = 1;
int total_iterations = 1;

int foo (int i) {
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &foo);
  int j = i*i;
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  int cap = MIN(num_threads,apex_get_thread_cap());
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
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &someThread);
#if defined (__APPLE__)
  //printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  //printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  printf("The scheduler ID of this thread: %d\n", *myid);
  while (total_iterations > 0) {
      if (*myid >= apex_get_thread_cap()) {
        apex_set_state(APEX_THROTTLED);
        //printf("Thread %d sleeping for a bit.\n", *myid);
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 1000000; // 1/1000 second
        // sleep a bit
        nanosleep(&tim , &tim2);
        apex_set_state(APEX_BUSY);
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
  apex_init(argv[0], 0, 1);
  num_threads = apex_hardware_concurrency();
  total_iterations = num_threads * ITERATIONS;
  apex_set_throttle_concurrency(true);
  apex_set_throttle_energy(true);

  apex_setup_timer_throttling(APEX_FUNCTION_ADDRESS, &foo, APEX_MINIMIZE_ACCUMULATED,
          APEX_ACTIVE_HARMONY, 1000000);

  int original_cap = apex_get_thread_cap();

  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &main);
  printf("PID of this process: %d\n", getpid());
  int i;
  pthread_t * thread = (pthread_t*)calloc(num_threads, sizeof(pthread_t));
  int * ids = (int*)calloc(num_threads, sizeof(int));
  for (i = 0 ; i < num_threads ; i++) {
    ids[i] = i;
    pthread_create(&(thread[i]), NULL, someThread, &(ids[i]));
  }
  for (i = 0 ; i < num_threads ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop(p);
  int final_cap = apex_get_thread_cap();
  if (final_cap < original_cap) {
      printf ("Test passed.\n");
  }
  apex_finalize();
  free(thread);
  return(0);
}

