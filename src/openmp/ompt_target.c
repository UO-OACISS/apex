#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "apex.h"

double start;
double end;
int main (void) {
  int sum = 0;
 	apex_init(__func__, 0, 1);
    apex_set_use_screen_output(1);
  start = omp_get_wtime();
  printf("Num devices available: %d\n",omp_get_num_devices() );

#pragma omp target parallel for map(tofrom:sum)
  for(int i = 0 ; i < 200000000; i++) {
    sum += 2;

}
  end = omp_get_wtime();
  printf ("time %f\n",(end-start));
  printf("sum = %d\n",sum);
    apex_finalize();
  return 0;
}
