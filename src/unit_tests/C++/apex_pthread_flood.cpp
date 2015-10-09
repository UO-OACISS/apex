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
#define FLOOD_LEVEL 1000
#endif

class ApexProxy {
private:
  std::string _name;
  apex::profiler * p;
  bool stopped;
public:
  ApexProxy(const char * func, const char * file, int line);
  ApexProxy(apex_function_address fpointer);
  ~ApexProxy();
  void stop() { stopped = true; apex::stop(p); };
};

ApexProxy::ApexProxy(const char * func, const char * file, int line) : stopped(false) {
  std::ostringstream s;
  s << func << " [" << file << ":" << line << "]";
  _name = std::string(s.str());
  p = apex::start(_name);
}

ApexProxy::ApexProxy(apex_function_address fpointer) : stopped(false) {
  p = apex::start(fpointer);
}

ApexProxy::~ApexProxy() {
  if (!stopped) apex::stop(p);
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
  unsigned long * result = (unsigned long *)tmp;
  apex::register_thread("threadTest thread");
  int i = 0;
  unsigned long total = 0;
  { // only time this for loop
    ApexProxy proxy = ApexProxy((apex_function_address)someThread);
    for (i = 0 ; i < ITERATIONS ; i++) {
        apex::profiler * p = apex::start((apex_function_address)foo);
        total += foo(i);
        apex::stop(p);
    }
  }
  apex::exit_thread();
  *result = total;
  return NULL;
}

int main(int argc, char **argv)
{
  apex::init(argc, argv, NULL);
  unsigned numthreads = apex::hardware_concurrency();
  if (argc > 1) {
    numthreads = strtoul(argv[1],NULL,0);
  }
  apex::set_node_id(0);
  apex::apex_options::use_screen_output(true);
  sleep(1); // if we don't sleep, the proc_read thread won't have time to read anything.

  ApexProxy proxy = ApexProxy((apex_function_address)main);
  printf("PID of this process: %d\n", getpid());
  std::cout << "Expecting " << numthreads << " threads." << std::endl;
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * numthreads));
  unsigned long * results = (unsigned long *)malloc(sizeof(unsigned long) * numthreads);
  std::atomic<int> thread_count(0);
  for (unsigned f = 0 ; f < FLOOD_LEVEL ; f++) {
#ifdef APEX_HAVE_TAU
    if (thread_count >= 127) { break; }
#endif
    unsigned i;
    for (i = 0 ; i < numthreads ; i++) {
        pthread_create(&(thread[i]), NULL, someThread, &(results[i]));
        thread_count++;
#ifdef APEX_HAVE_TAU
        if (thread_count >= 127) { break; }
#endif
    }
    unsigned newbreak = i;
    for (i = 0 ; i < numthreads && i < newbreak ; i++) {
        pthread_join(thread[i], NULL);
    }
  }
  free(results);
  proxy.stop();
  apex::finalize();
  apex::cleanup();
  return(0);
}

