#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE     512*512*512
#define ITERATIONS     3

int run_cpu( int argc, char** argv ) {
  printf( "The total memory allocated is %7.3lf MB.\n",
          2.0*sizeof(double)*ARRAY_SIZE/1024/1024 );

  double* a          = NULL;
  double* b          = NULL;
  int     num_errors = 0;
  double  time       = 0;
  double  start_time = 0;
  double  scalar     = 8.0;
  int     iterations = ITERATIONS;
  double  iteration_time[ITERATIONS];

  a = (double *) malloc( sizeof(double)*ARRAY_SIZE );
  b = (double *) malloc( sizeof(double)*ARRAY_SIZE );

  // initialize on the host
#pragma omp parallel for
for (size_t j=0; j<ARRAY_SIZE; j++)
    {
      a[j] = 0.0;
      b[j] = j;
    }

 start_time = omp_get_wtime();

  for(size_t i=0;i<iterations;i++)
    {
      iteration_time[i] = omp_get_wtime();
#pragma omp parallel for
      for (size_t j=0; j<ARRAY_SIZE; j++) {
        a[j] = a[j]+scalar*b[j];
      }
      iteration_time[i] = omp_get_wtime() - iteration_time[i];
    }

  time = omp_get_wtime()-start_time;

  printf("Time (s): %lf\n", time);

  // error checking
  for (size_t j=0; j<ARRAY_SIZE; j++) {
      if( fabs(a[j] - (double)j*iterations*scalar) > 0.000001  ) {
      num_errors++;
    }
    }

  free(a);
  free(b);

  if( num_errors == 0 ) printf( "Success!\n" );

  assert(num_errors == 0);

  return 0;
}

int run_gpu( int argc, char** argv )
{
  printf( "The total memory allocated is %7.3lf MB.\n",
          2.0*sizeof(double)*ARRAY_SIZE/1024/1024 );

  double* a          = NULL;
  double* b          = NULL;
  int     num_errors = 0;
  double  time       = 0;
  double  start_time = 0;
  double  scalar     = 8.0;
  int     iterations = ITERATIONS;
  double  iteration_time[ITERATIONS];

  a = (double *) malloc( sizeof(double)*ARRAY_SIZE );
  b = (double *) malloc( sizeof(double)*ARRAY_SIZE );

  // initialize on the host
#pragma omp parallel for
for (size_t j=0; j<ARRAY_SIZE; j++)
    {
      a[j] = 0.0;
      b[j] = j;
    }

#pragma omp target enter data map(to:a[0:ARRAY_SIZE])
#pragma omp target enter data map(to:b[0:ARRAY_SIZE])

 start_time = omp_get_wtime();

  for(size_t i=0;i<iterations;i++)
    {
      iteration_time[i] = omp_get_wtime();
#pragma omp target teams distribute parallel for
      for (size_t j=0; j<ARRAY_SIZE; j++) {
        a[j] = a[j]+scalar*b[j];
      }
      iteration_time[i] = omp_get_wtime() - iteration_time[i];
    }

  time = omp_get_wtime()-start_time;

#pragma omp target update from(a[0:ARRAY_SIZE])

  printf("Time (s): %lf\n", time);

  // error checking
  for (size_t j=0; j<ARRAY_SIZE; j++) {
      if( fabs(a[j] - (double)j*iterations*scalar) > 0.000001  ) {
      num_errors++;
    }
    }

#pragma omp target exit data map(release:a[0:ARRAY_SIZE])
#pragma omp target exit data map(release:b[0:ARRAY_SIZE])

  free(a);
  free(b);

  if( num_errors == 0 ) printf( "Success!\n" );

  assert(num_errors == 0);

  return 0;
}


int main( int argc, char** argv ) {
    run_cpu(argc, argv);
    run_gpu(argc, argv);
}