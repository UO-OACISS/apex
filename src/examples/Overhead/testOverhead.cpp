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

#define SPLIT 131072
#define ITERATIONS 1048576
#define INNER_ITERATION 4096

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
  UNUSED(tmp);
  apex::register_thread("threadTest thread");
  int i = 0;
  unsigned long total = 0;
  { // only time this for loop
    apex::profiler * st = apex::start((apex_function_address)someThread);
    for (i = 0 ; i < ITERATIONS ; i++) {
        apex::profiler * p = apex::start((apex_function_address)foo);
        total += foo(i);
        apex::stop(p);
        if (i % SPLIT == 0) {
            printf("t"); fflush(stdout);
        }
    }
    apex::stop(st);
  }
#if defined (__APPLE__)
  printf("%lu computed %lu (timed)\n", (unsigned long)pthread_self(), total);
#else
  printf("%u computed %lu (timed)\n", (unsigned int)pthread_self(), total);
#endif
  apex::exit_thread();
  return (void*)total;
}

void* someUntimedThread(void* tmp)
{
  UNUSED(tmp);
  apex::register_thread("threadTest thread");
  int i = 0;
  unsigned long total = 0;
  { // only time this for loop
    apex::profiler * sut = apex::start((apex_function_address)someUntimedThread);
    for (i = 0 ; i < ITERATIONS ; i++) {
        total += foo(i);
        if (i % SPLIT == 0) {
            printf("u"); fflush(stdout);
        }
    }
    apex::stop(sut);
  }
#if defined (__APPLE__)
  printf("%lu computed %lu (untimed)\n", (unsigned long)pthread_self(), total);
#else
  printf("%u computed %lu (untimed)\n", (unsigned int)pthread_self(), total);
#endif
  apex::exit_thread();
  return (void*)total;
}


int main(int argc, char **argv)
{
  apex::init(argv[0], 0, 1);
  apex::apex_options::use_screen_output(true);
  unsigned numthreads = apex::hardware_concurrency();
  if (argc > 1) {
    numthreads = strtoul(argv[1],NULL,0);
  }
  sleep(1); // if we don't sleep, the proc_read thread won't have time to read anything.

  apex::profiler * m = apex::start((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  std::cout << "Expecting " << numthreads << " threads." << std::endl;
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * numthreads));
  unsigned i;
  int timed = 0;
  int untimed = 0;
  for (i = 0 ; i < numthreads ; i++) {
    if (i % 2 == 0) {
      pthread_create(&(thread[i]), NULL, someUntimedThread, NULL);
      untimed++;
    } else {
      pthread_create(&(thread[i]), NULL, someThread, NULL);
      timed++;
    }
  }
  for (i = 0 ; i < numthreads ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex::stop(m);
  apex::finalize();
  apex_profile * without = apex::get_profile((apex_function_address)&someUntimedThread);
  apex_profile * with = apex::get_profile((apex_function_address)&someThread);
  apex_profile * footime = apex::get_profile((apex_function_address)&foo);
#ifdef APEX_USE_CLOCK_TIMESTAMP
#define METRIC " nanoseconds"
#else
#define METRIC " cycles"
#endif
  if (without) {
    double mean = without->accumulated/without->calls;
    double variance = ((without->sum_squares / without->calls) - (mean * mean));
    double stddev = sqrt(variance);
    std::cout << "Without timing: " << mean;
    std::cout << "±" << stddev << METRIC;
    std::cout << std::endl;
  }
  if (with) {
    double mean = with->accumulated/with->calls;
    double variance = ((with->sum_squares / with->calls) - (mean * mean));
    double stddev = sqrt(variance);
    std::cout << "   With timing: " << mean;
    std::cout << "±" << stddev << METRIC;
    std::cout << std::endl;
  }
  if (footime) {
    std::cout << "Total calls to 'foo': " << numthreads*ITERATIONS << std::endl;
    std::cout << "Timed calls to 'foo': " << (int)footime->calls << std::endl;
  }
  double overhead_per_call = 0.0;
  if (with && without && footime) {
    overhead_per_call = (with->accumulated - without->accumulated) / footime->calls;
    double percent_increase = (with->accumulated / without->accumulated) - 1.0;
    double foo_per_call = footime->accumulated / footime->calls;
    std::cout << "Estimated overhead per timer: ";
#ifdef APEX_USE_CLOCK_TIMESTAMP
    std::cout << overhead_per_call*1.0e9;
    std::cout << METRIC << " (" << percent_increase*100.0 << 
        "%), per call time in foo: " << (foo_per_call*1.0e9) << METRIC << std::endl;
#else
    std::cout << overhead_per_call;
    std::cout << METRIC << " (" << percent_increase*100.0 << 
        "%), per call time in foo: " << foo_per_call << METRIC << std::endl;
#endif
  }
  apex::cleanup();
  return(0);
}

