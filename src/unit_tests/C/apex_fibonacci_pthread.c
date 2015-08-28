#include "pthread.h"
#include <stdio.h>
#include <stdlib.h>
#include "apex.h"

#define THREADSTACK  1024*32

#define FIB_RESULTS_PRE 41
long long fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

void * fib (void * in) {
    apex_register_thread("fib thread");
	apex_profiler_handle p = apex_start(APEX_FUNCTION_ADDRESS, &fib);
    long long x = (long long)(in);
    if (x == 0) {
	    apex_stop(p);
        return (void*)0;
    }
    else if (x == 1) {
	    apex_stop(p);
        return (void*)1;
    }
    long long a = x-1;
    long long result_a = 0;
	pthread_attr_t attr_a; 
	pthread_attr_init(&attr_a);
	pthread_attr_setstacksize(&attr_a, THREADSTACK);
	pthread_t thread_a;
	pthread_create(&thread_a,&attr_a,fib,(void*)a);
    pthread_attr_destroy(&attr_a);

    long long b = x-2;
    long long result_b = 0;
	pthread_attr_t attr_b; 
	pthread_attr_init(&attr_b);
	pthread_attr_setstacksize(&attr_b, PTHREAD_STACK_MIN);
	pthread_t thread_b;
	pthread_create(&thread_b,&attr_b,fib,(void*)b);
    pthread_attr_destroy(&attr_a);

	pthread_join(thread_a,(void*)(&result_a));	
	pthread_join(thread_b,(void*)(&result_b));	
    
	apex_stop(p);
	apex_exit_thread();
    return (void*)(result_a + result_b);
}

long long main(long long argc, char *argv[]) {
    apex_init("apex_fibonacci_pthreads unit test");
	long long i = 10;

	if (argc != 2) {
		fprintf(stderr,"usage: pthreads <integer value>\n");
		fprintf(stderr,"Using default value of 10\n");
	} else {
	    i = atol(argv[1]);
	}

	if (i < 1) {
		fprintf(stderr,"%lu must be>= 1\n", i);
		return -1;
	}

	long long result = (long long)fib((void*)i);
    printf("fib of %lu is %lu (valid value: %lu)\n", i, result, fib_results[i]);
	apex_finalize();
}

