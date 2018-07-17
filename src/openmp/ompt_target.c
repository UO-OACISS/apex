#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

double start;
double end;
int main (void) {
  int sum = 0;
  start = omp_get_wtime();
  printf("Num devices available: %d\n",omp_get_num_devices() );

#pragma omp target parallel for map(tofrom:sum)
  for(int i = 0 ; i < 2000000000; i++) {
    sum += 2;

}
  end = omp_get_wtime();
  printf ("time %f\n",(end-start));
  printf("sum = %d\n",sum);
  return 0;
}
