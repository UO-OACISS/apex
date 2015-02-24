#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include "apex.h"
#include "apex_throttling.h"

#define NUM_THREADS 48
#define ITERATIONS 10000
#define SLEEPY_TIME 1000000 // 1,000,000

int foo (int i) {
  apex_profiler_handle p = apex_start_address((apex_function_address)foo);
  int j = i*i;
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  tim.tv_nsec = (unsigned long)(SLEEPY_TIME * randval * apex_get_thread_cap());
  nanosleep(&tim , &tim2);
  apex_stop_profiler(p);
  return j;
}

typedef void*(*start_routine_t)(void*);

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  int *myid = (int*)tmp;
  apex_register_thread("threadTest thread");
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  apex_profiler_handle p = apex_start_address((apex_function_address)someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  printf("The scheduler ID of this thread: %d\n", *myid);
  int i = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
      if (apex_throttleOn && *myid >= apex_get_thread_cap()) {
        //printf("Thread %d sleeping for a bit.\n", *myid);
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 500000000; // 1/2 second
        // sleep a bit
        nanosleep(&tim , &tim2);
      } else {
	    foo(i);
      }
  }
  apex_stop_profiler(p);
  return NULL;
}

int main(int argc, char **argv)
{
  apex_init_args(argc, argv, NULL);
  apex_set_node_id(0);

  apex_setup_throttling((apex_function_address)foo);

  apex_profiler_handle p = apex_start_address((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
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
  apex_stop_profiler(p);
  apex_finalize();
  return(0);
}

