#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include "apex.h"

/* Example taken from:
   https://computing.llnl.gov/tutorials/openMP/#REDUCTION
*/

int main (int argc, char** argv) {
	int   i, n;
 	double a[100], b[100], result;
    apex_set_use_screen_output(1);

 	/* Some initializations */
 	n = 100;
 	result = 0.0;
 	for (i=0; i < n; i++) {
   		a[i] = i * 1.0;
   		b[i] = i * 2.0;
   	}

#pragma omp parallel
    {
#pragma omp for
   	    for (i=0; i < n; i++) {
     	    result = result + (a[i] * b[i]);
        }
#pragma omp barrier
    }

 	printf("Final result= %f\n",result);

    return 0;
}

