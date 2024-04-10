#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>
#include <climits>
#include <thread>
#include <chrono>
#include "utils.hpp"
#include <atomic>

#define ITERATIONS 1
#define INNER_ITERATION 1024*16
#ifdef APEX_HAVE_TAU
#define FLOOD_LEVEL 15 // TAU has a limit of 128 threads.
#else
#define FLOOD_LEVEL 100
#endif

inline int foo (int i) {
  int j;
  int dummy = 1;
  for (j = 0 ; j < INNER_ITERATION ; j++) {
    dummy = dummy * (dummy + i);
    if (dummy > (INT_MAX >> 1)) {
      dummy = 1;
    }
  }
  return dummy;
}

typedef void*(*start_routine_t)(void*);

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  apex::scoped_thread mythread("threadTest thread");
  unsigned long * result = (unsigned long *)tmp;
  int i = 0;
  unsigned long total = 0;
  { // only time this for loop
    apex::scoped_timer proxy((apex_function_address)someThread);
    for (i = 0 ; i < ITERATIONS ; i++) {
        apex::profiler * p = apex::start((apex_function_address)foo);
        total += foo(i);
        apex::stop(p);
    }
  }
  *result = total;
  return NULL;
}

int main(int argc, char **argv)
{
  apex::init(argv[0], 0, 1);
  unsigned test_numthreads = apex::hardware_concurrency();
  if (argc > 1) {
    test_numthreads = strtoul(argv[1],NULL,0);
  }
  apex::apex_options::use_screen_output(true);

  apex::scoped_timer proxy(__func__);
  printf("PID of this process: %d\n", getpid());
  std::cout << "Expecting " << test_numthreads << " threads." << std::endl;
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * test_numthreads));
  unsigned long * results = (unsigned long *)malloc(sizeof(unsigned long) * test_numthreads);
  std::atomic<int> thread_count(0);
  for (unsigned f = 0 ; f < FLOOD_LEVEL ; f++) {
#ifdef APEX_HAVE_TAU
    if (thread_count >= 120) { break; }
#endif
    unsigned i;
    for (i = 0 ; i < test_numthreads ; i++) {
        pthread_create(&(thread[i]), NULL, someThread, &(results[i]));
        thread_count++;
#ifdef APEX_HAVE_TAU
        if (thread_count >= 120) { break; }
#endif
    }
    unsigned newbreak = i;
    for (i = 0 ; i < test_numthreads && i < newbreak ; i++) {
        pthread_join(thread[i], NULL);
    }
  }
  free(results);
  proxy.stop();
  apex::finalize();
  apex::cleanup();
  free(thread);
  return(0);
}

