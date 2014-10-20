#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex.hpp>
#include <sstream>

#define NUM_THREADS 8
#define ITERATIONS 1000000

class ApexProxy {
private:
  std::string _name;
  void * profiler;
public:
  ApexProxy(const char * func, const char * file, int line);
  ApexProxy(void *fpointer);
  ~ApexProxy();
};

ApexProxy::ApexProxy(const char * func, const char * file, int line) {
  std::ostringstream s;
  s << func << " [" << file << ":" << line << "]";
  _name = std::string(s.str());
  profiler = apex::start(_name);
}

ApexProxy::ApexProxy(void *fpointer) {
  profiler = apex::start(fpointer);
}

ApexProxy::~ApexProxy() {
  apex::stop(profiler);
};

int foo (int i) {
  ApexProxy proxy = ApexProxy((void*)foo);
  return i*i;
}

typedef void*(*start_routine_t)(void*);

void* someThread(void* tmp)
{
  apex::register_thread("threadTest thread");
  //ApexProxy proxy = ApexProxy(__func__, __FILE__, __LINE__);
  ApexProxy proxy = ApexProxy((void*)someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  int i = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
	  foo(i);
  }
  return NULL;
}

int main(int argc, char **argv)
{
  apex::init(argc, argv, NULL);
  apex::set_node_id(0);
  const apex_event_type when = STOP_EVENT;
  apex::register_periodic_policy(1000000, [](apex_context const& context){
       void * foo_addr = (void*)(foo);
       apex::profile * p = apex::profiler_listener::get_profile(foo_addr);
       if (p != NULL) {
           cout << "Periodic: " << foo_addr << " " << p->get_calls() << " " << p->get_mean() << " seconds." << endl;
       }
       return true;
  });
  apex::register_policy(when, [](apex_context const& context){
       static __thread unsigned int not_all_the_time = 0;
       if (++not_all_the_time % 500000 != 0) return true; // only do 2 out of a million
       void * foo_addr = (void*)(foo);
       apex::profile * p = apex::profiler_listener::get_profile(foo_addr);
       if (p != NULL) {
           cout << "Event: " << foo_addr << " " << p->get_calls() << " " << p->get_mean() << " seconds." << endl;
       }
       return true;
  });
  ApexProxy proxy = ApexProxy((void*)main);
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex::finalize();
  return(0);
}

