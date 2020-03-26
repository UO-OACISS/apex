#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <apex_api.hpp>

#define NUM_THREADS 8
#define ITERATIONS 50000

apex_event_type custom_type_1;
apex_event_type custom_type_2;

int foo (int i) {
  static __thread apex::profiler * my_profiler;
  int result = 0;

  if (i % 2 == 0) {
    // do start/yield/start/stop
    my_profiler = apex::start((apex_function_address)&foo);
    result = i*i; // work
    apex::yield(my_profiler);
    result += i*i; // work
    my_profiler = apex::start((apex_function_address)&foo);
    result += i*i; // work
    apex::stop(my_profiler);
  } else {
    // do start/stop/resume/stop
    my_profiler = apex::start((apex_function_address)&foo);
    result = i*i; // work
    apex::stop(my_profiler);
    result += i*i; // work
    my_profiler = apex::resume((apex_function_address)&foo);
    result += i*i; // work
    apex::stop(my_profiler);
  }

  return result;
}

void* someThread(void* tmp)
{
  APEX_UNUSED(tmp);
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
    APEX_UNUSED(context);
    apex_profile * p = apex::get_profile((apex_function_address)&foo);
    if (p != NULL) {
        printf("Periodic Policy: %p %d %f seconds.\n", (void*)foo, (int)p->calls, p->accumulated/p->calls);
    }
    return APEX_NOERROR;
}

int startup_policy(apex_context const context) {
    APEX_UNUSED(context);
    printf("Startup Policy...\n");
    return APEX_NOERROR;
}

int main(int argc, char **argv)
{
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex_register_periodic_policy unit test", 0, 1);
  apex::apex_options::use_screen_output(true);
  apex_policy_handle * on_periodic = apex::register_periodic_policy(100000, policy_periodic);
  apex::profiler* my_profiler = apex::start(__func__);
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  if (on_periodic != nullptr) {
      //printf("Deregistering %d...\n", on_periodic->id);
      //apex::deregister_policy(on_periodic);
      apex::dump(false);
      apex::stop_all_async_threads();
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

