#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>
#include <iostream>
#include <climits>
#include <atomic>

#define NUM_THREADS 8
#define NUM_ITERATIONS 100

#define UNUSED(x) (void)(x)

std::atomic<uint64_t> func_count(0);
std::atomic<uint64_t> yield_count(0);

uint64_t do_work(uint64_t work) {
  apex::profiler * p = apex::start((apex_function_address)(do_work));
  int i;
  uint64_t dummy = 1;
  for (i = 0 ; i < 1234567 ; i++) {
    dummy = dummy * (dummy + work);
    if (dummy > (INT_MAX >> 1)) {
      dummy = 1;
    }
  }
  func_count++;
  if (dummy % 2 == 0) {
    apex::stop(p);
  } else {
    yield_count++;
    apex::yield(p);
  }
  return dummy;
}


void* someThread(void* tmp)
{
  UNUSED(tmp);
  apex::self_stopping_timer proxy((apex_function_address)someThread, "threadTest thread");
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  int i;
  for (i = 0 ; i < NUM_ITERATIONS ; i++) {
    do_work(i);
  }
  return NULL;
}


int main(int argc, char **argv)
{
  apex::init(argv[0], 0, 1);
  apex::self_stopping_timer proxy((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  std::cout << "Function calls : " << func_count << std::endl;
  std::cout << "Yields         : " << yield_count << std::endl;
  std::cout << "Value Expected : " << (func_count - yield_count) << std::endl;
  apex_profile * profile = apex::get_profile((apex_function_address)(do_work));
  if (profile) {
    std::cout << "Value Reported : " << profile->calls << std::endl;
    if ((func_count - yield_count) == profile->calls) { 
        std::cout << "Test passed." << std::endl;
    } else if ((func_count - yield_count) > profile->calls) { 
	    // OK to under-report.
        std::cout << "Test passed." << std::endl;
	}
  }
  proxy.stop();
  apex::finalize();
  apex::cleanup();
  return(0);
}

