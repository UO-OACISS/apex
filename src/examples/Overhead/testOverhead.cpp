#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>
#include <climits>
#include <thread>

#define ITERATIONS 1024*128
#define INNER_ITERATION 1024*4

class ApexProxy {
private:
  std::string _name;
  apex::profiler * p;
public:
  ApexProxy(const char * func, const char * file, int line);
  ApexProxy(apex_function_address fpointer);
  ~ApexProxy();
};

ApexProxy::ApexProxy(const char * func, const char * file, int line) {
  std::ostringstream s;
  s << func << " [" << file << ":" << line << "]";
  _name = std::string(s.str());
  p = apex::start(_name);
}

ApexProxy::ApexProxy(apex_function_address fpointer) {
  p = apex::start(fpointer);
}

ApexProxy::~ApexProxy() {
  apex::stop(p);
};

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
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  ApexProxy proxy = ApexProxy((apex_function_address)someThread);
  //printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  int i = 0;
  unsigned long total = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
    apex::profiler * p = apex::start((apex_function_address)foo);
    total += foo(i);
    apex::stop(p);
  }
  printf("%u computed %lu (timed)\n", (unsigned int)pthread_self(), total);
  return NULL;
}

void* someUntimedThread(void* tmp)
{
  UNUSED(tmp);
  apex::register_thread("threadTest thread");
  ApexProxy proxy = ApexProxy((apex_function_address)someUntimedThread);
  //printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  int i = 0;
  unsigned long total = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
	  total += foo(i);
  }
  printf("%u computed %lu (untimed)\n", (unsigned int)pthread_self(), total);
  return NULL;
}


int main(int argc, char **argv)
{
  apex::init(argc, argv, NULL);
  apex::set_node_id(0);
  sleep(1); // if we don't sleep, the proc_read thread won't have time to read anything.

  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  ApexProxy proxy = ApexProxy((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  unsigned numthreads = std::thread::hardware_concurrency();
  pthread_t thread[numthreads];
  int i;
  for (i = 0 ; i < numthreads ; i++) {
    pthread_create(&(thread[i]), NULL, someUntimedThread, NULL);
  }
  for (i = 0 ; i < numthreads ; i++) {
    pthread_join(thread[i], NULL);
  }
  for (i = 0 ; i < numthreads ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < numthreads ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex::finalize();
  apex_profile * with = apex::get_profile((apex_function_address)&someThread);
  apex_profile * without = apex::get_profile((apex_function_address)&someUntimedThread);
  apex_profile * footime = apex::get_profile((apex_function_address)&foo);
  apex_profile * mhz = apex::get_profile(std::string("cpuinfo.0:cpu MHz"));
  std::cout << "Without timing: " << without->accumulated/without->calls;
  std::cout << ", with timing: " << with->accumulated/with->calls << std::endl;
  std::cout << "Expected calls to 'foo': " << numthreads*ITERATIONS;
  std::cout << ", timed calls to 'foo': " << (int)footime->calls << std::endl;
  double percall = (with->accumulated - footime->accumulated) / footime->calls;
  double milliseconds = percall * 1.0e3;
  double microseconds = percall * 1.0e6;
  double nanoseconds = percall * 1.0e9;
  //std::cout << "Overhead per timer: " << milliseconds << " ms" << std::endl;
  //std::cout << "Overhead per timer: " << microseconds << " us" << std::endl;
  std::cout << "Overhead per timer: " << nanoseconds << " ns" << std::endl;
  if (mhz) {
    double cycles = percall * mhz->accumulated * 1.0e6;
    std::cout << "Overhead per timer: " << cycles << " cycles" << std::endl;
  }
  return(0);
}

