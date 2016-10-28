#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include "apex.h"

#define NUM_THREADS 3

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  UNUSED(tmp);
  apex_register_thread("threadTest thread");
  apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS,(void*)&someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  apex_stop(profiler);
  apex_exit_thread();
  return NULL;
}

int main(int argc, char **argv)
{
  apex_init("apex_exit_thread unit test", 0, 1);
  apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS,(void*)&main);
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * NUM_THREADS));
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  apex_stop(profiler);
  apex_finalize();
  return(0);
}

