#include <unistd.h>
#include <stdio.h>
#include <omp.h>

int main (int argc, char** argv) {
#pragma omp parallel
    {
        printf("Hello from thread %d of %d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
        fflush(stdout);
    }
    return 0;
}

