#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <apex_api.hpp>

#define NUM_THREADS 8
#define ITERATIONS 10

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
  apex::register_thread("threadTest thread");
  apex::sample_value("some value", 42);
  apex::profiler* my_profiler = apex::start((apex_function_address)&someThread);
  int i = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
      foo(i);
  }
  apex::stop(my_profiler);
  apex::exit_thread();
  return NULL;
}

int startup_policy(apex_context const context) {
    printf("Startup Policy...\n");
    return APEX_NOERROR;
}

int main(int argc, char **argv)
{
  std::set<apex_event_type> when = {APEX_STARTUP, APEX_SHUTDOWN, APEX_NEW_NODE, APEX_NEW_THREAD,
      APEX_START_EVENT, APEX_STOP_EVENT, APEX_SAMPLE_VALUE};
  std::set<apex_policy_handle*> handles = 
  apex::register_policy(when, [](apex_context const& context)->int{
    switch(context.event_type) {
      case APEX_STARTUP: std::cout      << "Startup event\n"; fflush(stdout); break;
      case APEX_SHUTDOWN: std::cout     << "Shutdown event\n"; fflush(stdout); break;
      case APEX_NEW_NODE: std::cout     << "New node event\n"; fflush(stdout); break;
      case APEX_NEW_THREAD: std::cout   << "New thread event\n"; fflush(stdout); break;
      case APEX_START_EVENT: std::cout  << "Start event\n"; fflush(stdout); break;
      case APEX_STOP_EVENT: std::cout   << "Stop event\n"; fflush(stdout); break;
      case APEX_SAMPLE_VALUE: std::cout << "Sample value event\n"; fflush(stdout); break;
      default: std::cout << "Unknown event" << std::endl;
    }
    return APEX_NOERROR;
  });

  apex::apex_options::use_screen_output(true);

  apex::init("apex_register_policy_set unit test", 0, 1);

  apex::profiler* my_profiler = apex::start((apex_function_address)&main);
  pthread_t thread[NUM_THREADS];
  int i;
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

