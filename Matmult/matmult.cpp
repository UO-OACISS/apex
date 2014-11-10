/******************************************************************************
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. 
*
*   Modified from here:
*   https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
*
*   For  PAPI_FP_INS, the exclusive count for the event: 
*   for (null) [OpenMP location: file:matmult.c ]
*   should be  2E+06 / Number of Threads 
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "matmult_initialize.h"

#ifdef PTHREADS
#include <pthread.h>
#endif /* PTHREADS */

#include "apex.hpp"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 512
#endif

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

double** allocateMatrix(int rows, int cols) {
  //void * profiler = apex::start((void*)(allocateMatrix));
  void * profiler = apex::start(__func__);
  int i;
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  apex::stop(profiler);
  return matrix;
}

#ifdef APP_USE_INLINE_MULTIPLY
__inline double multiply(double a, double b) {
	return a * b;
}
#endif /* APP_USE_INLINE_MULTIPLY */

// cols_a and rows_b are the same value
void compute_nested(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //void * profiler = apex::start((void*)(compute_nested));
  void * profiler = apex::start(__func__);
  int i,j,k;
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
    for (i=0; i<rows_a; i++) {
      {
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else 
          c[i][j] += a[i][k] * b[k][j];
#endif 
        }
      }
      }
    }
  }   /*** End of parallel region ***/
  apex::stop(profiler);
}

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //void * profiler = apex::start((void*)(compute));
  void * profiler = apex::start(__func__);
  int i,j,k;
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
    for (i=0; i<rows_a; i++) {
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
        }
      }
    }
  }   /*** End of parallel region ***/
  apex::stop(profiler);
}

void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //void * profiler = apex::start((void*)(compute_interchange));
  void * profiler = apex::start(__func__);
  int i,j,k;
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
        }
      }
    }
  }   /*** End of parallel region ***/
  apex::stop(profiler);
}

double do_work(void) {
  //void * profiler = apex::start((void*)(do_work));
  void * profiler = apex::start(__func__);
  double **a,           /* matrix A to be multiplied */
  **b,           /* matrix B to be multiplied */
  **c;           /* result matrix C */
  a = allocateMatrix(NRA, NCA);
  b = allocateMatrix(NCA, NCB);
  c = allocateMatrix(NRA, NCB);  

/*** Spawn a parallel region explicitly scoping all variables ***/

  initialize(a, NRA, NCA);
  initialize(b, NCA, NCB);
  initialize(c, NRA, NCB);

  compute(a, b, c, NRA, NCA, NCB);
  compute_interchange(a, b, c, NRA, NCA, NCB);

  apex::stop(profiler);
  return c[0][1]; 
}

void * threaded_func(void *data)
{
  apex::register_thread("threaded_func");
  do_work();
  return NULL;
}

int main (int argc, char *argv[]) 
{
  apex::init(argc, argv, NULL);
  apex::set_node_id(0);
  //void * profiler = apex::start((void*)(main));
  void * profiler = apex::start(__func__);

#ifdef PTHREADS
  int ret;
  pthread_attr_t  attr;
  pthread_t       tid1, tid2, tid3;
#endif /* PTHREADS */

#ifdef PTHREADS
  if (ret = pthread_create(&tid1, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (1) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 1...\n");

  if (ret = pthread_create(&tid2, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (2) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 2...\n");

  if (ret = pthread_create(&tid3, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (3) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 3...\n");

#endif /* PTHREADS */

/* On thread 0: */
  do_work();

#ifdef PTHREADS 
  if (ret = pthread_join(tid1, NULL) )
  {
    printf("Error: pthread_join (1) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_join(tid2, NULL) )
  {
    printf("Error: pthread_join (2) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_join(tid3, NULL) )
  {
    printf("Error: pthread_join (3) fails ret = %d\n", ret);
    exit(1);
  }   
#endif /* PTHREADS */

  printf ("Done.\n");
  apex::stop(profiler);
  apex::finalize();
  return 0;
}

