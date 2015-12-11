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

#define ITERATIONS 1024*32
#define INNER_ITERATION 1024*8

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
  apex::init(argc, argv, NULL);
  unsigned numthreads = apex::hardware_concurrency();
  if (argc > 1) {
    numthreads = strtoul(argv[1],NULL,0);
  }
  apex::set_node_id(0);
  sleep(1); // if we don't sleep, the proc_read thread won't have time to read anything.

  apex::profiler * m = apex::start((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  std::cout << "Expecting " << numthreads << " threads." << std::endl;
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * numthreads));
  unsigned i;
  for (i = 0 ; i < numthreads ; i++) {
    if (i % 2 == 0) {
      pthread_create(&(thread[i]), NULL, someUntimedThread, NULL);
    } else {
      pthread_create(&(thread[i]), NULL, someThread, NULL);
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
  apex_profile * mhz = apex::get_profile(std::string("cpuinfo.0:cpu MHz"));
  if (without) {
    double mean = without->accumulated/without->calls;
    double variance = ((without->sum_squares / without->calls) - (mean * mean));
    double stddev = sqrt(variance);
    std::cout << "Without timing: " << mean;
    std::cout << "±" << stddev << std::endl;;
  }
  if (with) {
    double mean = with->accumulated/with->calls;
    double variance = ((with->sum_squares / with->calls) - (mean * mean));
    double stddev = sqrt(variance);
    std::cout << ", with timing: " << mean;
    std::cout << "±" << stddev << std::endl;
  }
  if (footime) {
    std::cout << "Expected calls to 'foo': " << numthreads*ITERATIONS;
    std::cout << ", timed calls to 'foo': " << (int)footime->calls << std::endl;
  }
  double percall1 = 0.0;
  if (with && without && footime) {
    percall1 = (with->accumulated - without->accumulated) / (numthreads * ITERATIONS);
    double percent = (with->accumulated / without->accumulated) - 1.0;
    double foopercall = footime->accumulated / footime->calls;
    //double percall2 = (with->accumulated - footime->accumulated) / (numthreads * ITERATIONS);
    int nanoseconds1 = percall1 * 1.0e6;
    int nanofoo = foopercall * 1.0e6;
    //int nanoseconds2 = percall2 * 1.0e9;
    std::cout << "Average overhead per timer: ";
    std::cout << nanoseconds1;
    std::cout << " cycles (" << percent*100.0 << "%), per call time in foo: " << nanofoo << " cycles " << std::endl;
    //std::cout << "Overhead (2) per timer: ";
    //std::cout << nanoseconds2;
    //std::cout << " ns" << std::endl;
  }
  if (mhz && percall1 > 0.0) {
    double cycles1 = percall1 * mhz->accumulated * 1.0e6;
    //double cycles2 = percall2 * mhz->accumulated * 1.0e6;
    std::cout << "Overhead (1) per timer: " << cycles1 << " cycles" << std::endl;
    //std::cout << "Overhead (2) per timer: " << cycles2 << " cycles" << std::endl;
  }
  apex::cleanup();
  return(0);
}

