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
#include <unistd.h>
#include <omp.h>
#include <math.h>

int fib(int n) {
  int x,y;
  if (n<2) return n;
  #pragma omp task untied shared(x)
  { x = fib(n-1); }
  #pragma omp task untied shared(y)
  { y = fib(n-2); }
  #pragma omp taskwait
  return x+y;
}

int fibouter(int n) {
  int answer = 0;
  #pragma omp parallel shared(answer)
  {
    #pragma omp single 
    {
      #pragma omp task shared(answer) 
      {
	    answer = fib(n);
      }
    }
  }
  return answer;
}

int main (int argc, char *argv[]) 
{
  printf("Main...\n");
  fflush(stdout);
  printf ("\nDoing forking tasks..."); fflush(stdout);
  fibouter(10);

  printf ("Done.\n");
  fflush(stdout);

  return 0;
}

