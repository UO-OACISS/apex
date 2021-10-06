#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include "apex.h"

#define NUM_THREADS 100

#define UNUSED(x) (void)(x)

apex_profiler_handle handles[NUM_THREADS];

void* someThread(void* tmp)
{
    uintptr_t myid = (uintptr_t)tmp;
    apex_register_thread("threadTest thread");
    apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS,(const void*)&someThread);
    // Even-numbered threads start the timers
    if (myid % 2 == 0) {
        handles[myid] = apex_start(APEX_NAME_STRING,"cross-thread timer");
        sleep(1);
    }
    printf("PID of this process: %d\n", getpid());
#if defined (__APPLE__)
    printf("The ID of this thread is: %lu\n", (unsigned long)pthread_self());
#else
    printf("The ID of this thread is: %u\n", (unsigned int)pthread_self());
#endif
    sleep(1);
    // Odd-numbered threads stop the timers
    if (myid % 2 == 1) {
        apex_stop(handles[myid-1]);
    }
    apex_stop(p);
    apex_exit_thread();
    return NULL;
}

int main(int argc, char **argv)
{
    apex_set_use_screen_output(1);
    apex_init("apex_exit_thread unit test", 0, 1);
    apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS,(const void*)&main);
    pthread_t * thread = (pthread_t*)(malloc(sizeof(pthread_t) * NUM_THREADS));
    uintptr_t i;
    for (i = 0 ; i < NUM_THREADS ; i++) {
        pthread_create(&(thread[i]), NULL, someThread, (void*)i);
    }
    for (i = 0 ; i < NUM_THREADS ; i++) {
        pthread_join(thread[i], NULL);
    }
    apex_stop(profiler);
    apex_finalize();
    apex_cleanup();
    free(thread);
    return(0);
}

