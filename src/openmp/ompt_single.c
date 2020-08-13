#include <omp.h>
#include <stdio.h>
#include "apex.h"

int main (void) {
	int a, i;
    apex_set_use_screen_output(1);
#pragma omp parallel shared(a) private(i)
	{
#pragma omp master
  		a = 0;
  		// To avoid race conditions, add a barrier here.
#pragma omp for reduction(+:a)
  		for (i = 0; i < 10; i++) { a += i; }
#pragma omp single
  		printf ("Sum is %d\n", a);
	}
}