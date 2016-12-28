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

int policy_event(apex_context const context) {
    switch (context.event_type) {
        case APEX_STARTUP:
        {
            printf("Startup event.\n");
            break;
        }
        case APEX_SHUTDOWN:
        {
            printf("Shutdown event.\n");
            break;
        }
        case APEX_NEW_NODE:
        {
            printf("New Node event.\n");
            break;
        }
        case APEX_NEW_THREAD:
        {
            printf("New Thread event.\n");
            break;
        }
        case APEX_START_EVENT:
        {
            printf("Start event.\n");
            break;
        }
        case APEX_RESUME_EVENT:
        {
            printf("Resume event.\n");
            break;
        }
        case APEX_STOP_EVENT:
        {
            printf("Stop event.\n");
            break;
        }
        case APEX_YIELD_EVENT:
        {
            printf("Yield event.\n");
            break;
        }
        case APEX_SAMPLE_VALUE:
        {
            printf("Sample Value event.\n");
            break;
        }
        case APEX_PERIODIC:
        {
            printf("Periodic event.\n");
            break;
        }
        case APEX_CUSTOM_EVENT_1:
        {
            printf("Custom event 1.\n");
            break;
        }
        default:
        {
            if (context.event_type < APEX_MAX_EVENTS) {
                printf("Custom event 2.\n");
            } else {
                printf("Unknown event type!\n");
                exit(-1);
            }
            break;
        }
    }
    return APEX_NOERROR;
}

int startup_policy(apex_context const context) {
    printf("Startup Policy...\n");
    return APEX_NOERROR;
}

int main(int argc, char **argv)
{
  apex_policy_handle * on_startup = apex::register_policy(APEX_STARTUP, startup_policy);
  apex::register_policy(APEX_SHUTDOWN, policy_event);
  apex_policy_handle * on_new_node = apex::register_policy(APEX_NEW_NODE, policy_event);
  apex_policy_handle * on_new_thread = apex::register_policy(APEX_NEW_THREAD, policy_event);

  apex::init("apex_register_custom_event unit test", 0, 1);

  apex_policy_handle * on_start_event = apex::register_policy(APEX_START_EVENT, policy_event);
  apex_policy_handle * on_stop_event = apex::register_policy(APEX_STOP_EVENT, policy_event);
  apex_policy_handle * on_resume_event = apex::register_policy(APEX_RESUME_EVENT, policy_event);
  apex_policy_handle * on_yield_event = apex::register_policy(APEX_YIELD_EVENT, policy_event);
  apex_policy_handle * on_sample_value = apex::register_policy(APEX_SAMPLE_VALUE, policy_event);
  custom_type_1 = apex::register_custom_event("CUSTOM 1");
  custom_type_2 = apex::register_custom_event("CUSTOM 2");
  apex_policy_handle * on_custom_event_1 = apex::register_policy(custom_type_1, policy_event);
  apex_policy_handle * on_custom_event_2 = apex::register_policy(custom_type_2, policy_event);

  apex::profiler* my_profiler = apex::start((apex_function_address)&main);
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  // now un-register the policies 
  if (on_startup != nullptr) {
      printf("Deregistering %d...\n", on_startup->id);
      apex::deregister_policy(on_startup);
  }
  /*
  if (on_shutdown != nullptr) {
      printf("Deregistering %d...\n", on_shutdown->id);
      apex::deregister_policy(on_shutdown);
  }
  */
  if (on_new_node != nullptr) {
      printf("Deregistering %d...\n", on_new_node->id);
      apex::deregister_policy(on_new_node);
  }
  if (on_new_thread != nullptr) {
      printf("Deregistering %d...\n", on_new_thread->id);
      apex::deregister_policy(on_new_thread);
  }
  if (on_start_event != nullptr) {
      printf("Deregistering %d...\n", on_start_event->id);
      apex::deregister_policy(on_start_event);
  }
  if (on_stop_event != nullptr) {
      printf("Deregistering %d...\n", on_stop_event->id);
      apex::deregister_policy(on_stop_event);
  }
  if (on_resume_event != nullptr) {
      printf("Deregistering %d...\n", on_resume_event->id);
      apex::deregister_policy(on_resume_event);
  }
  if (on_yield_event != nullptr) {
      printf("Deregistering %d...\n", on_yield_event->id);
      apex::deregister_policy(on_yield_event);
  }
  if (on_sample_value != nullptr) {
      printf("Deregistering %d...\n", on_sample_value->id);
      apex::deregister_policy(on_sample_value);
  }
  if (on_custom_event_1 != nullptr) {
      printf("Deregistering %d...\n", on_custom_event_1->id);
      apex::deregister_policy(on_custom_event_1);
  }
  if (on_custom_event_2 != nullptr) {
      printf("Deregistering %d...\n", on_custom_event_2->id);
      apex::deregister_policy(on_custom_event_2);
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

