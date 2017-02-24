#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include "apex_api.hpp"

void* someThread(void* tmp)
{
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker thread %d", *tid);
    /* Register this thread with APEX */
    apex::register_thread(name);
    /* Start a timer */
    apex::profiler* p = apex::start((apex_function_address)&someThread);
    /* ... */
    /* do some computation */
    /* ... */
    /* stop the timer */
    apex::stop(p);
    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
    /* start a timer */
    apex::profiler* p = apex::start("main");
    /* Spawn two threads */
    pthread_t thread[2];
    int tid = 0;
    pthread_create(&(thread[0]), NULL, someThread, &tid);
    int tid2 = 1;
    pthread_create(&(thread[1]), NULL, someThread, &tid2);
    /* wait for the threads to finish */
    pthread_join(thread[0], NULL);
    pthread_join(thread[1], NULL);
    /* stop our main timer */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
    return 0;
}

