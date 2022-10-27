#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main( int argc, char** argv )
{

  double*   a = NULL;
  double*   b = NULL;
  double scalar = 8.0;
  int num_errors = 0;
  int num_elements = 1024;

  printf( "Number of devices: %d\n", omp_get_num_devices() );

  #pragma omp target
  {
    if( !omp_is_initial_device() )
      printf( "Hello world from accelerator.\n" );
    else
      printf( "Hello world from host.\n" );
  }


/*
  #pragma omp target teams thread_limit(5)
  #pragma omp distribute parallel for simd
  for (int i = 0 ; i < 1000 ; i++) {
    printf("Hi from thread %d of %d threads in team %d of %d teams is using index %d\n",
    omp_get_thread_num(),
    omp_get_num_threads(),
    omp_get_team_num(),
    omp_get_num_teams(), i);
  }
*/
  a = (double *) malloc( sizeof(double)*num_elements );
  b = (double *) malloc( sizeof(double)*num_elements );

  // initialize on the host
  #pragma omp parallel for
  for (size_t j=0; j<num_elements; j++)
    {
      a[j] = 0.0;
      b[j] = j;
    }

  //#pragma omp parallel for
  #pragma omp target teams distribute parallel for simd map(tofrom:a[:num_elements]) map(to:b[:num_elements])
  for (size_t j=0; j<num_elements; j++) {
    a[j] += scalar*b[j];
  }

  // error checking
  #pragma omp parallel for reduction(+:num_errors)
  for (size_t j=0; j<num_elements; j++) {
    if( fabs(a[j] - (double)j*scalar) > 0.000001  ) {
      num_errors++;
    }
  }

  free(a);
  free(b);

  if(num_errors == 0) printf( "Success!\n" );

  assert(num_errors == 0);

  return 0;
}
