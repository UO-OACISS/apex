#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex.h>

#define NUM_THREADS 8
#define ITERATIONS 50000

int foo (int i) {
#ifdef __APPLE__
  apex_profiler_handle my_profiler = apex_start(APEX_NAME_STRING, "foo");
#else
  apex_profiler_handle my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &foo);
#endif
  int result = i*i;
  apex_stop(my_profiler);
  return result;
}

typedef void*(*start_routine_t)(void*);

void* someThread(void* tmp)
{
  apex_register_thread("threadTest thread");
#ifdef __APPLE__
  apex_profiler_handle my_profiler = apex_start(APEX_NAME_STRING, "someThread");
#else
  apex_profiler_handle my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &someThread);
#endif
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
  printf("done in foo\n");
  apex_stop(my_profiler);
  apex_exit_thread();
  printf("done in someThread\n");
  return NULL;
}

int policy_periodic(apex_context const context) {
#ifdef __APPLE__
    apex_profile * p = apex_get_profile(APEX_NAME_STRING, "foo");
    if (p != NULL) {
        printf("Periodic Policy: 'foo' %d %f seconds.\n", (int)p->calls, p->accumulated/p->calls);
    }
#else
    apex_profile * p = apex_get_profile(APEX_FUNCTION_ADDRESS, &foo);
    if (p != NULL) {
        printf("Periodic Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
#endif
    return APEX_NOERROR;
}

int policy_event(apex_context const context) {
    static __thread unsigned int not_every_time = 0;
    if (not_every_time++ % 500000 != 0) return APEX_NOERROR;
#ifdef __APPLE__
    apex_profile * p = apex_get_profile(APEX_NAME_STRING, "foo");
    if (p != NULL) {
        printf("Event Policy: 'foo' %d %f seconds.\n", (int)p->calls, p->accumulated/p->calls);
    }
#else
    apex_profile * p = apex_get_profile(APEX_FUNCTION_ADDRESS, &foo);
    if (p != NULL) {
        printf("Event Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
#endif
    return APEX_NOERROR;
}

int main(int argc, char **argv)
{
  apex_init(argv[0], 0, 1);
  apex_set_use_policy(true);
  apex_set_use_screen_output(true);
  apex_set_use_profile_output(true);
  const apex_event_type when = APEX_STOP_EVENT;
  apex_policy_handle * on_periodic = apex_register_periodic_policy(1000000, policy_periodic);
  apex_policy_handle * on_event = apex_register_policy(when, policy_event);
#ifdef __APPLE__
  apex_profiler_handle my_profiler = apex_start(APEX_NAME_STRING, "main");
#else
  apex_profiler_handle my_profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
#endif
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
    printf("Joined thread %d\n", i);
  }
  // now un-register the policies 
  apex_deregister_policy(on_periodic);
  apex_deregister_policy(on_event);

  printf("Running without policies now...\n");
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
    printf("Joined thread %d\n", i);
  }
  apex_stop(my_profiler);
  apex_finalize();
  apex_cleanup();
  return(0);
}

