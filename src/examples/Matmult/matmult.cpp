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

#define PTHREADS 1

#ifdef PTHREADS
#include <pthread.h>
#endif /* PTHREADS */

#include "apex_api.hpp"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 512
#endif

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

double** allocateMatrix(int rows, int cols) {
  //apex::profiler* p = apex::start((void*)(allocateMatrix));
  apex::profiler* p = apex::start(__func__);
  int i;
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  apex::stop(p);
  return matrix;
}

void freeMatrix(double** matrix, int rows) {
  apex::profiler* p = apex::start(__func__);
  int i;
  for (i=0; i<rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
  apex::stop(p);
}

#ifdef APP_USE_INLINE_MULTIPLY
__inline double multiply(double a, double b) {
  return a * b;
}
#endif /* APP_USE_INLINE_MULTIPLY */

// cols_a and rows_b are the same value
void compute_nested(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //apex::profiler* p = apex::start((void*)(compute_nested));
  apex::profiler* p = apex::start(__func__);
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
  apex::stop(p);
}

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //apex::profiler* p = apex::start((void*)(compute));
  apex::profiler* p = apex::start(__func__);
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
  apex::stop(p);
}

void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  //apex::profiler* p = apex::start((void*)(compute_interchange));
  apex::profiler* p = apex::start(__func__);
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
  apex::stop(p);
}

double do_work(void) {
  //apex::profiler* p = apex::start((void*)(do_work));
  apex::profiler* p = apex::start(__func__);
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

  double result = c[0][1];
  freeMatrix(c, NRA);
  freeMatrix(b, NCA);
  freeMatrix(a, NRA);
  apex::stop(p);
  return result;
}

#define UNUSED(x) (void)(x)

void * threaded_func(void *data)
{
  UNUSED(data);
  apex::register_thread("threaded_func");
  do_work();
  apex::exit_thread();
  return NULL;
}

int main (int argc, char *argv[]) 
{
  apex::init(argv[0], 0, 1);
  //apex::profiler* p = apex::start((void*)(main));
  apex::profiler* p = apex::start(__func__);

#ifdef PTHREADS
  int ret;
  pthread_t       tid1, tid2, tid3;
#endif /* PTHREADS */

#ifdef PTHREADS
  if ((ret = pthread_create(&tid1, NULL, threaded_func, NULL) ))
  {
    printf("Error: pthread_create (1) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 1...\n");

  if ((ret = pthread_create(&tid2, NULL, threaded_func, NULL) ))
  {
    printf("Error: pthread_create (2) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 2...\n");

  if ((ret = pthread_create(&tid3, NULL, threaded_func, NULL) ))
  {
    printf("Error: pthread_create (3) fails ret = %d\n", ret);
    exit(1);
  }   
  printf("Spawned thread 3...\n");

#endif /* PTHREADS */

/* On thread 0: */
  do_work();

#ifdef PTHREADS 
  if ((ret = pthread_join(tid1, NULL) ))
  {
    printf("Error: pthread_join (1) fails ret = %d\n", ret);
    exit(1);
  }   

  if ((ret = pthread_join(tid2, NULL) ))
  {
    printf("Error: pthread_join (2) fails ret = %d\n", ret);
    exit(1);
  }   

  if ((ret = pthread_join(tid3, NULL) ))
  {
    printf("Error: pthread_join (3) fails ret = %d\n", ret);
    exit(1);
  }   
#endif /* PTHREADS */

  printf ("Done.\n");
  apex::stop(p);
  apex::finalize();
  return 0;
}

