#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include "apex.h"

__attribute__((noinline)) void foo(void) {
#pragma omp parallel
    {
        printf("Hello from thread %d of %d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
        fflush(stdout);
    }
}

__attribute__((noinline)) void bar(void) {
#pragma omp parallel
    {
        printf("Hello again from thread %d of %d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
        fflush(stdout);
    }
}

int main (int argc, char** argv) {
    apex_set_use_screen_output(1);
    foo();
    bar();
    return 0;
}

