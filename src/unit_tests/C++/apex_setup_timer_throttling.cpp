#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include "apex_api.hpp"

#define NUM_THREADS 8
#define ITERATIONS 5000
#define SLEEPY_TIME 10000 // 10,000

int total_iterations = NUM_THREADS * ITERATIONS;

using namespace apex;
using namespace std;

int foo (int i) {
  profiler* p = start((apex_function_address)foo);
  int j = i*i;
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  int cap = min(NUM_THREADS,get_thread_cap());
  tim.tv_nsec = (unsigned long)(SLEEPY_TIME * randval * (cap * cap));
  nanosleep(&tim , &tim2);
  stop(p);
  return j;
}

typedef void*(*start_routine_t)(void*);

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  int *myid = (int*)tmp;
  register_thread("threadTest thread");
  profiler* p = start((apex_function_address)someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  printf("The scheduler ID of this thread: %d\n", *myid);
  while (total_iterations > 0) {
      if (*myid >= get_thread_cap()) {
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
            printf("%d iterations left, cap is %d\n", total_iterations, get_thread_cap());
        }
      }
  }
  printf("Thread done: %d. Current Cap: %d.\n", *myid, get_thread_cap());
  stop(p);
  exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  init(argv[0], 0, 1);
  apex_options::throttle_concurrency(1);
  //print_options();

  setup_timer_throttling((apex_function_address)foo, APEX_MAXIMIZE_THROUGHPUT,
          APEX_DISCRETE_HILL_CLIMBING, 1000000);
  int original_cap = get_thread_cap();

  profiler* p = start((apex_function_address)main);
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
  stop(p);
  int final_cap = get_thread_cap();
  if (final_cap < original_cap) {
    std::cout << "Test passed." << std::endl;
  }
  finalize();
  shutdown_throttling();
  return(0);
}

