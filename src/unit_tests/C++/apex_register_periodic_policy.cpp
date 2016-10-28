#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <apex_api.hpp>

#define NUM_THREADS 8
#define ITERATIONS 10

apex_event_type custom_type_1;
apex_event_type custom_type_2;

int foo (int i) {
  static __thread apex::profiler * my_profiler;
  if (i % 2 == 0) {
    my_profiler = apex::start((apex_function_address)&foo);
  } else {
    my_profiler = apex::resume((apex_function_address)&foo);
  }
  int result = i*i;
  if (i % 2 == 0) {
    apex::yield(my_profiler);
  } else {
    apex::stop(my_profiler);
  }
  return result;
}

void* someThread(void* tmp)
{
  apex::register_thread("threadTest thread");
  apex::custom_event(custom_type_1, NULL);
  apex::sample_value("some value", 42);
  apex::profiler* my_profiler = apex::start((apex_function_address)&someThread);
  int i = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
      foo(i);
  }
  apex::custom_event(custom_type_2, NULL);
  apex::stop(my_profiler);
  apex::exit_thread();
  return NULL;
}

int policy_periodic(apex_context const context) {
    apex_profile * p = apex::get_profile((apex_function_address)&foo);
    if (p != NULL) {
        printf("Periodic Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
    return APEX_NOERROR;
}

int startup_policy(apex_context const context) {
    printf("Startup Policy...\n");
    return APEX_NOERROR;
}

int main(int argc, char **argv)
{
  apex::init("apex_register_periodic_policy unit test", 0, 1);
  apex_policy_handle * on_periodic = apex::register_periodic_policy(1000000, policy_periodic);
  apex::profiler* my_profiler = apex::start((apex_function_address)&main);
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  if (on_periodic != nullptr) {
      printf("Deregistering %d...\n", on_periodic->id);
      apex::deregister_policy(on_periodic);
  }

  printf("Running without policies now...\n");
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex::stop(my_profiler);
  apex::finalize();
  return(0);
}

