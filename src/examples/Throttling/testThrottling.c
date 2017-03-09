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
#define SLEEPY_TIME 1000 // 1000

int numcores = 8;
int total_iterations = ITERATIONS;
bool test_passed = false;
int original_cap = 8;

int foo (int i) {
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &foo);
  int j = i*i;
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  int cap = MIN(numcores,apex_get_thread_cap());
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
  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  printf("The scheduler ID of this thread: %d\n", *myid);
  int my_total_iterations = __sync_fetch_and_add(&(total_iterations), 0);
  while (my_total_iterations > 0) {
      if (*myid >= apex_get_thread_cap()) {
        //printf("Thread %d sleeping for a bit.\n", *myid);
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 100000000; // 1/10 second
        // sleep a bit
        nanosleep(&tim , &tim2);
      } else {
        foo(my_total_iterations);
        my_total_iterations = __sync_fetch_and_sub(&(total_iterations),1);
        if (my_total_iterations % 1000 == 0) {
            printf("%d iterations left, cap is %d\n", my_total_iterations, apex_get_thread_cap());
        }
      }
      my_total_iterations = __sync_fetch_and_add(&(total_iterations), 0);
  }
  printf("Thread done: %d. Current Cap: %d.\n", *myid, apex_get_thread_cap());
  apex_stop(p);
  apex_exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  apex_init(argv[0], 0, 1);
  apex_set_throttle_concurrency(true);
  apex_set_throttle_energy(true);

  apex_setup_timer_throttling(APEX_FUNCTION_ADDRESS, &foo, APEX_MINIMIZE_ACCUMULATED,
          APEX_DISCRETE_HILL_CLIMBING, 100000);

  original_cap = apex_get_thread_cap();

  apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &main);
  //printf("PID of this process: %d\n", getpid());
  numcores = sysconf(_SC_NPROCESSORS_ONLN);
  total_iterations = total_iterations * numcores;
  original_cap = numcores;
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * numcores));
  int * ids = (int*)(malloc(sizeof(int) * numcores));
  int i;
  for (i = 0 ; i < numcores ; i++) {
    ids[i] = i;
    pthread_create(&(thread[i]), NULL, someThread, &(ids[i]));
  }
  for (i = 0 ; i < numcores ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop(p);
  int final_cap = apex_get_thread_cap();
  if (test_passed || (final_cap <= original_cap) || (original_cap <= 4)) {
    printf("Test passed.\n");
  }
  apex_finalize();
  free(thread);
  free(ids);
  return(0);
}

