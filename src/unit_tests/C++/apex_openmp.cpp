#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "apex.h"

#define N 1024*1024
#define MAX_THREADS 256

#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__ ((aligned(x)))
#else
#define ALIGNED_(x) __declspec(align(x))
#endif

double no_sharing(double* x, double* y)
{
  double sum=0.0;
  ALIGNED_(128) double sum_local[MAX_THREADS] = {0};
  #pragma omp parallel
  {
   int me = omp_get_thread_num();

   int i;
   #pragma omp for
   for (i = 0; i < N; i++) {
     sum_local[me] = sum_local[me] + (x[i] * y[i]);
   }

   #pragma omp atomic
   sum += sum_local[me];
  }
  return sum;
}

void my_init(double* x)
{
  double randval = 1.0 + (((double)(rand())) / RAND_MAX);
  #pragma omp parallel
  {
   int i;
   #pragma omp for
   for (i = 0; i < N; i++) {
     x[i] = randval;
   }
  }
}

int main(int argc, char** argv)
{
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  static double x[N];
  static double y[N];
  printf("Initializing...\n"); fflush(stdout);
  my_init(x);
  my_init(y);

  double result = 0.0;

  printf("No Sharing...\n"); fflush(stdout);
  result = no_sharing(x, y);
  printf("Result: %f\n", result);

  apex_finalize();
  return 0;
}
