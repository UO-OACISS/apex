#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <apex_api.hpp>
#include <sstream>

#define NUM_THREADS 3

#define UNUSED(x) (void)(x)

void* someThread(void* tmp)
{
  UNUSED(tmp);
  apex::scoped_thread ast("threadTest thread");
  apex::scoped_timer proxy((uint64_t)&someThread);
  printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
  printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
  printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
  return NULL;
}


int main(int argc, char **argv)
{
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init(NULL, 0, 1);
  apex::scoped_timer proxy(__func__);
  double currentpower = apex::current_power_high();
  printf("Power at start: %f Watts\n", currentpower);
  printf("PID of this process: %d\n", getpid());
  pthread_t thread[NUM_THREADS];
  int i;
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_create(&(thread[i]), NULL, someThread, NULL);
  }
  for (i = 0 ; i < NUM_THREADS ; i++) {
    pthread_join(thread[i], NULL);
  }
  currentpower = apex::current_power_high();
  printf("Power at end: %f Watts\n", currentpower);
  proxy.stop();
  apex::finalize();
  return(0);
}

