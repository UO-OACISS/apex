#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex.h>

#define NUM_THREADS 8
#define ITERATIONS 1000000

int foo (int i) {
  apex_profiler_handle my_profiler = apex_start_address(foo);
  int result = i*i;
  apex_stop_profiler(my_profiler);
  return result;
}

typedef void*(*start_routine_t)(void*);

void* someThread(void* tmp)
{
  apex_register_thread("threadTest thread");
  apex_profiler_handle my_profiler = apex_start_address((void*)someThread);
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
  apex_stop_profiler(my_profiler);
  return NULL;
}

int policy_periodic(apex_context const context) {
    apex_profile * p = apex_get_profile_from_address(foo);
    if (p != NULL) {
        printf("Periodic Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
    return 1;
}

int policy_event(apex_context const context) {
    static __thread unsigned int not_every_time = 0;
    if (not_every_time++ % 500000 != 0) return 1;
    apex_profile * p = apex_get_profile_from_address(foo);
    if (p != NULL) {
        printf("Event Policy: %p %d %f seconds.\n", foo, (int)p->calls, p->accumulated/p->calls);
    }
    return 1;
}

int main(int argc, char **argv)
{
  apex_init_args(argc, argv, NULL);
  apex_set_use_policy(true);
  apex_set_use_screen_output(true);
  apex_set_use_profile_output(true);
  apex_set_node_id(0);
  const apex_event_type when = STOP_EVENT;
  apex_register_periodic_policy(1000000, policy_periodic);
  apex_register_policy(when, policy_event);
  apex_profiler_handle my_profiler = apex_start_address((void*)main);
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop_profiler(my_profiler);
  apex_finalize();
  return(0);
}

