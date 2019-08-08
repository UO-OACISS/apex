#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "apex.h"

#define MAX_ITERATIONS 10
#define N 4096*4096
#define MAX_THREADS 256

#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__ ((aligned(x)))
#else
#define ALIGNED_(x) __declspec(align(x))
#endif

double openmp_reduction(double* x, double* y)
{
  double sum=0.0;
  #pragma omp parallel
  {
   int i;
   #pragma omp for reduction( + : sum ) schedule(runtime)
   for (i = 0; i < N; i++) {
     #pragma omp atomic
     sum += (x[i] * y[i]);
   }

  }
  return sum;
}

double false_sharing(double* x, double* y)
{
  double sum=0.0;
  double sum_local[MAX_THREADS] = {0};
  #pragma omp parallel
  {
   int me = omp_get_thread_num();

   int i;
   #pragma omp for schedule(runtime)
   for (i = 0; i < N; i++) {
     sum_local[me] = sum_local[me] + (x[i] * y[i]);
   }

   #pragma omp atomic
   sum += sum_local[me];
  }
  return sum;
}

double no_sharing(double* x, double* y)
{
  double sum=0.0;
  ALIGNED_(128) double sum_local[MAX_THREADS] = {0};
  #pragma omp parallel
  {
   int me = omp_get_thread_num();

   int i;
   #pragma omp for schedule(runtime)
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
   #pragma omp for schedule(runtime)
   for (i = 0; i < N; i++) {
     x[i] = randval;
   }
  }
}

int main(int argc, char** argv)
{
  static double x[N];
  static double y[N];
  apex_init("openmp test", 0, 1);
  printf("Initializing x...\n"); fflush(stdout);
  my_init(x);
  printf("Initializing y...\n"); fflush(stdout);
  my_init(y);

  double result = 0.0;
  int i = 0;

  for (i = 0 ; i < MAX_ITERATIONS ; i++) {
  	printf("%d Reduction sharing... ", i); fflush(stdout);
  	result = openmp_reduction(x, y);
  	printf("%d Result: %f\n", i, result);

  	printf("%d False sharing... ", i); fflush(stdout);
  	result = false_sharing(x, y);
  	printf("%d Result: %f\n", i, result);

  	printf("%d No Sharing... ", i); fflush(stdout);
  	result = no_sharing(x, y);
  	printf("%d Result: %f\n", i, result); fflush(stdout);
  }

  apex_finalize();
  return 0;
}
