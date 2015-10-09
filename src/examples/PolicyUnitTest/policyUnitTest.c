#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <apex.h>

#define NUM_THREADS 8
#define ITERATIONS 10

int custom_type_1;
int custom_type_2;

int foo (int i) {
  static __thread apex_profiler_handle my_profiler;
  if (i % 2 == 0) {
    my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &foo);
  } else {
    my_profiler = apex_resume(APEX_FUNCTION_ADDRESS, &foo);
  }
  int result = i*i;
  if (i % 2 == 0) {
    apex_yield(my_profiler);
  } else {
    apex_stop(my_profiler);
  }
  return result;
}

void* someThread(void* tmp)
{
  apex_register_thread("threadTest thread");
  apex_custom_event(custom_type_1, NULL);
  apex_sample_value("some value", 42);
  apex_profiler_handle my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &someThread);
  int i = 0;
  for (i = 0 ; i < ITERATIONS ; i++) {
      foo(i);
  }
  apex_custom_event(custom_type_2, NULL);
  apex_stop(my_profiler);
  apex_exit_thread();
  return NULL;
}

int policy_periodic(apex_context const context) {
    apex_profile * p = apex_get_profile(APEX_FUNCTION_ADDRESS, &foo);
    if (p != NULL) {
        printf("Periodic Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
    return APEX_NOERROR;
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
        case APEX_CUSTOM_EVENT_2:
        {
            printf("Custom event 2.\n");
            break;
        }
        default:
        {
            printf("Unknown event type!\n");
            exit(-1);
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
  apex_policy_handle * on_startup = apex_register_policy(APEX_STARTUP, startup_policy);
  apex_policy_handle * on_shutdown = apex_register_policy(APEX_SHUTDOWN, policy_event);
  apex_policy_handle * on_new_node = apex_register_policy(APEX_NEW_NODE, policy_event);
  apex_policy_handle * on_new_thread = apex_register_policy(APEX_NEW_THREAD, policy_event);

  apex_init_args(argc, argv, NULL);
  apex_set_node_id(0);

  apex_policy_handle * on_start_event = apex_register_policy(APEX_START_EVENT, policy_event);
  apex_policy_handle * on_stop_event = apex_register_policy(APEX_STOP_EVENT, policy_event);
  apex_policy_handle * on_resume_event = apex_register_policy(APEX_RESUME_EVENT, policy_event);
  apex_policy_handle * on_yield_event = apex_register_policy(APEX_YIELD_EVENT, policy_event);
  apex_policy_handle * on_sample_value = apex_register_policy(APEX_SAMPLE_VALUE, policy_event);
  custom_type_1 = apex_register_custom_event("CUSTOM 1");
  custom_type_2 = apex_register_custom_event("CUSTOM 2");
  apex_policy_handle * on_custom_event_1 = apex_register_policy(custom_type_1, policy_event);
  apex_policy_handle * on_custom_event_2 = apex_register_policy(custom_type_2, policy_event);

  apex_policy_handle * on_periodic = apex_register_periodic_policy(1000000, policy_periodic);

  apex_profiler_handle my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  // now un-register the policies 
  printf("Deregistering %d...\n", on_startup->id);
  printf("Deregistering %d...\n", on_shutdown->id);
  printf("Deregistering %d...\n", on_new_node->id);
  printf("Deregistering %d...\n", on_new_thread->id);
  printf("Deregistering %d...\n", on_start_event->id);
  printf("Deregistering %d...\n", on_stop_event->id);
  printf("Deregistering %d...\n", on_resume_event->id);
  printf("Deregistering %d...\n", on_yield_event->id);
  printf("Deregistering %d...\n", on_sample_value->id);
  printf("Deregistering %d...\n", on_custom_event_1->id);
  printf("Deregistering %d...\n", on_custom_event_2->id);
  printf("Deregistering %d...\n", on_periodic->id);
  apex_deregister_policy(on_startup);
  apex_deregister_policy(on_shutdown);
  apex_deregister_policy(on_new_node);
  apex_deregister_policy(on_new_thread);
  apex_deregister_policy(on_start_event);
  apex_deregister_policy(on_stop_event);
  apex_deregister_policy(on_resume_event);
  apex_deregister_policy(on_yield_event);
  apex_deregister_policy(on_sample_value);
  apex_deregister_policy(on_custom_event_1);
  apex_deregister_policy(on_custom_event_2);
  apex_deregister_policy(on_periodic);

  printf("Running without policies now...\n");
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop(my_profiler);
  apex_finalize();
  return(0);
}

