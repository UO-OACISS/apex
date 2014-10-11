#include "matmult_initialize.h"
#include "apex.hpp"

void initialize(double **matrix, int rows, int cols) {
  //void * profiler = apex::start((void*)(initialize));
  void * profiler = apex::start(__func__);
  int i,j;
  {
    /*** Initialize matrices ***/
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        matrix[i][j]= i+j;
      }
    }
  }
  apex::stop(profiler);
}

