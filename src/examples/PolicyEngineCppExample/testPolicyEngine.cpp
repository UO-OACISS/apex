#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <apex_types.h>
#include <sstream>

#define NUM_THREADS 8
#define ITERATIONS 50000

using namespace std;

int foo (int i) {
  apex::profiler * profiler = apex::start((apex_function_address)foo);
  int j = i*i;
  apex::stop(profiler);
  return j;
}

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  UNUSED(tmp);
  apex::register_thread("threadTest thread");
  apex::profiler * profiler = apex::start((apex_function_address)someThread);
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
  apex::stop(profiler);
  apex::exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  APEX_UNUSED(argc);
  apex::init(argv[0], 0, 1);
  apex::apex_options::use_policy(true);
  apex::apex_options::use_screen_output(true);
  apex::apex_options::use_profile_output(true);
  apex::profiler * profiler = apex::start(__func__);
  const apex_event_type when = APEX_STOP_EVENT;
  apex::register_periodic_policy(1000000, [](apex_context const& context){
       UNUSED(context);
       apex_function_address foo_addr = (apex_function_address)(foo);
       apex_profile * p = apex::get_profile(foo_addr);
       if (p != NULL) {
           cout << "Periodic: " << foo_addr << " " << p->calls << " " << p->accumulated/p->calls << " seconds." << endl;
       }
       return APEX_NOERROR;
  });
  apex::register_policy(when, [](apex_context const& context)->int{
       UNUSED(context);
       static APEX_NATIVE_TLS unsigned int not_all_the_time = 0;
       if (++not_all_the_time % 500000 != 0) return APEX_NOERROR; // only do 2 out of a million
       apex_function_address foo_addr = (apex_function_address)(foo);
       apex_profile * p = apex::get_profile(foo_addr);
       if (p != NULL) {
           cout << "Event: " << foo_addr << " " << p->calls << " " << p->accumulated/p->calls << " seconds." << endl;
       }
       return APEX_NOERROR;
  });
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex::stop(profiler);
  apex::finalize();
  return(0);
}

