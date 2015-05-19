#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>
#include <iostream>
#include <climits>
#include <boost/atomic.hpp>

#define NUM_THREADS 8
#define NUM_ITERATIONS 100

class ApexProxy {
private:
  std::string _name;
  apex::profiler * p;
  bool stopped;
public:
  ApexProxy(const char * func, const char * file, int line);
  ApexProxy(apex_function_address fpointer);
  ~ApexProxy();
  void stop(void) { apex::stop(p); stopped = true; };
};

ApexProxy::ApexProxy(const char * func, const char * file, int line) : stopped(false) {
  std::ostringstream s;
  s << func << " [" << file << ":" << line << "]";
  _name = std::string(s.str());
  p = apex::start(_name);
}

ApexProxy::ApexProxy(apex_function_address fpointer) {
  p = apex::start(fpointer);
}

ApexProxy::~ApexProxy() {
    if (!stopped) apex::stop(p);
};

#define UNUSED(x) (void)(x)

boost::atomic<uint64_t> func_count(0);
boost::atomic<uint64_t> yield_count(0);

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
  apex::register_thread("threadTest thread");
  ApexProxy proxy = ApexProxy((apex_function_address)someThread);
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  int i;
  for (i = 0 ; i < NUM_ITERATIONS ; i++) {
    do_work(i);
  }
  proxy.stop();
  apex::exit_thread();
  return NULL;
}


int main(int argc, char **argv)
{
  apex::init(argc, argv, NULL);
  apex::set_node_id(0);
  ApexProxy proxy = ApexProxy((apex_function_address)main);
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
    }
  }
  proxy.stop();
  apex::finalize();
  apex::cleanup();
  return(0);
}

