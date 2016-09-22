#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "pthread.h"
#include "apex.h"

#ifndef PTHREAD_STACK_MIN
#define PTHREAD_STACK_MIN 1024*16
#endif

#define FIB_RESULTS_PRE 41
int fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

typedef struct scratchpad {
    int x;
    int f_x;
} scratchpad_t;

void * fib (void * in) {
    apex_register_thread("fib thread");
    apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &fib);
    scratchpad_t* scratch = (scratchpad_t*)(in);
    if (scratch->x == 0) {
        scratch->f_x = 0;
        apex_stop(p);
        pthread_exit(NULL);
    }
    else if (scratch->x == 1) {
        scratch->f_x = 1;
        apex_stop(p);
        pthread_exit(NULL);
    }
    scratchpad_t a;
    a.x = (scratch->x)-1;
    a.f_x = 0;
    pthread_attr_t attr_a; 
    pthread_attr_init(&attr_a);
    //pthread_attr_setstacksize(&attr_a, PTHREAD_STACK_MIN);
    pthread_t thread_a;
    int rc_a = pthread_create(&thread_a, &attr_a, fib, (void*)&a);
    if (rc_a == EAGAIN) {
        //printf("Insufficient resources to create another thread. \n EAGAIN A system-imposed limit on the number of threads was encountered.  There are a number of limits that may trigger this error: the RLIMIT_NPROC soft resource limit (set via setrlimit(2)), which limits the number of processes and threads for a real user ID, was reached; the kernel's system- wide limit on the number of processes and threads, /proc/sys/kernel/threads-max, was reached (see proc(5)); or the maximum number of PIDs, /proc/sys/kernel/pid_max, was reached (see proc(5)).");
    } else if (rc_a == EINVAL) {
        //printf("Invalid settings in attr.");
    } else if (rc_a == EPERM) {
        //printf("No permission to set the scheduling policy and parameters specified in attr.");
    }
    pthread_attr_destroy(&attr_a);

    scratchpad_t b;
    b.x = (scratch->x)-2;
    b.f_x = 0;
    pthread_attr_t attr_b; 
    pthread_attr_init(&attr_b);
    //pthread_attr_setstacksize(&attr_b, PTHREAD_STACK_MIN);
    pthread_t thread_b;
    int rc_b = pthread_create(&thread_b,&attr_b,fib,(void*)&b);
       if (rc_b == EAGAIN) {
        //printf("Insufficient resources to create another thread. \n EAGAIN A system-imposed limit on the number of threads was encountered.  There are a number of limits that may trigger this error: the RLIMIT_NPROC soft resource limit (set via setrlimit(2)), which limits the number of processes and threads for a real user ID, was reached; the kernel's system- wide limit on the number of processes and threads, /proc/sys/kernel/threads-max, was reached (see proc(5)); or the maximum number of PIDs, /proc/sys/kernel/pid_max, was reached (see proc(5)).");
    } else if (rc_b == EINVAL) {
        //printf("Invalid settings in attr.");
    } else if (rc_b == EPERM) {
        //printf("No permission to set the scheduling policy and parameters specified in attr.");
    }
    pthread_attr_destroy(&attr_a);

    if (rc_a > 0) {
    // thread failed, just execute.
      fib((void*)&a);
      //a.f_x == fib_results[a.x];
    } else {
      pthread_join(thread_a,NULL);    
    }

    if (rc_b > 0) {
    // thread failed, just execute.
      fib((void*)&b);
      //b.f_x == fib_results[b.x];
    } else {
      pthread_join(thread_b,NULL);    
    }

    if (a.f_x != fib_results[a.x]) {
      printf("WRONG! fib of %d is NOT %d (valid value: %d)\n", a.x, a.f_x, fib_results[a.x]);
      a.f_x = fib_results[a.x];
    }
    if (b.f_x != fib_results[b.x]) {
      printf("WRONG! fib of %d is NOT %d (valid value: %d)\n", b.x, b.f_x, fib_results[b.x]);
      b.f_x = fib_results[b.x];
    }
    scratch->f_x = a.f_x + b.f_x;
    apex_stop(p);
    apex_exit_thread();
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    apex_init("apex_fibonacci_pthreads unit test");
#if defined(APEX_HAVE_TAU) || defined(DEBUG)
    int i = 5;
#else
    int i = 10;
#endif

    if (argc != 2) {
        fprintf(stderr,"usage: pthreads <integer value>\n");
        fprintf(stderr,"Using default value of %d\n", i);
    } else {
        i = atoi(argv[1]);
    }

    if (i < 1) {
        fprintf(stderr,"%d must be>= 1\n", i);
        return -1;
    }

    //int result = (int)fib((void*)i);
    scratchpad_t scratch;
    scratch.x = i;
    scratch.f_x = 0;

    pthread_attr_t attr; 
    pthread_attr_init(&attr);
    size_t oldStackSize;
    pthread_attr_getstacksize(&attr, &oldStackSize);
    //pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
    pthread_t thread;
    pthread_create(&thread,&attr,fib,(void*)&scratch);
    pthread_attr_destroy(&attr);
    printf("Default stack: %ld\n", oldStackSize);
    printf("Min stack: %d\n", PTHREAD_STACK_MIN);
    pthread_join(thread,NULL);
    printf("fib of %d is %d (valid value: %d)\n", i, scratch.f_x, fib_results[i]);
    apex_finalize();
    return 0;
}

