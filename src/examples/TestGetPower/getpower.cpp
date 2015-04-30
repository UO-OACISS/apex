#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>

#define NUM_THREADS 3

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

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  UNUSED(tmp);
  apex::register_thread("threadTest thread");
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  ApexProxy proxy = ApexProxy((apex_function_address)someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  //apex::finalize();
  return NULL;
}


int main(int argc, char **argv)
{
  apex::init(argc, argv, NULL);
  apex::set_node_id(0);
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  ApexProxy proxy = ApexProxy((apex_function_address)main);
  double currentpower = apex::current_power_high();
  printf("Power at start: %f Watts\n", currentpower);
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  someThread(NULL);
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  currentpower = apex::current_power_high();
  printf("Power at end: %f Watts\n", currentpower);
  apex::finalize();
  return(0);
}

