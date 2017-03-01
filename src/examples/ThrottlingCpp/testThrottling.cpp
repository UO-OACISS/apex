#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <atomic>

#include "apex_api.hpp"

#define ITERATIONS 1000
#define SLEEPY_TIME 1000 // 10000

const int NUM_THREADS = std::thread::hardware_concurrency();
std::atomic<int> total_iterations(NUM_THREADS * ITERATIONS);

using namespace apex;
using namespace std;

int foo (int i) {
  profiler* p = start((apex_function_address)foo);
  int j = i*i;
  //double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  struct timespec tim, tim2;
  tim.tv_sec = 0;
  // sleep just a bit longer, based on number of active threads.
  int cap = min(NUM_THREADS,get_thread_cap());
  //tim.tv_nsec = (unsigned long)(SLEEPY_TIME * randval * (cap * cap));
  tim.tv_nsec = (unsigned long)(SLEEPY_TIME * (cap * cap));
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
  int my_total_iterations = total_iterations;
  while (my_total_iterations > 0) {
      if (*myid >= get_thread_cap()) {
        //printf("Thread %d sleeping for a bit.\n", *myid);
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 100000000; // 1/10 second
        // sleep a bit
        nanosleep(&tim , &tim2);
      } else {
        foo(my_total_iterations);
        total_iterations--;
        if (my_total_iterations % 1000 == 0) {
            printf("%d iterations left, cap is %d\n", my_total_iterations, get_thread_cap());
        }
      }
      my_total_iterations = total_iterations;
  }
  printf("Thread done: %d. Current Cap: %d.\n", *myid, get_thread_cap());
  stop(p);
  exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  init(argv[0], 0, 1);
  //print_options();
  apex_options::throttle_concurrency(true);
  apex_options::throttle_energy(true);

  setup_timer_throttling((apex_function_address)foo, APEX_MINIMIZE_ACCUMULATED,
          APEX_DISCRETE_HILL_CLIMBING, 1000000);
  int original_cap = get_thread_cap();

  profiler* p = start((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  pthread_t* thread = new pthread_t[NUM_THREADS];
  int* ids = new int[NUM_THREADS];
  int i;
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
  delete [] thread;
  delete [] ids;
  return(0);
}

