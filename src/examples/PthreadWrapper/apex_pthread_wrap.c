#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#define NUM_THREADS 2

#define UNUSED(x) (void)(x)

// This thread exits with return
void* someThread(void* tmp)
{
  UNUSED(tmp);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  return NULL;
}

// This thread exits with pthread_exit
void* someOtherThread(void* tmp)
{
  UNUSED(tmp);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  pthread_exit(NULL);
}

int main(int argc, char **argv)
{
  apex_set_use_screen_output(1);
  pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * NUM_THREADS));
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    if (i % 2 == 0) {
      pthread_create(&(thread[i]), NULL, someThread, NULL);
    } else {
      pthread_create(&(thread[i]), NULL, someOtherThread, NULL);
    }
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  return(0);
}

