#include "matmult_initialize.h"
#include "apex_api.hpp"

void initialize(double **matrix, int rows, int cols) {
  //profiler* p = apex::start((void*)(initialize));
  apex::profiler* p = apex::start(__func__);
  int i,j;
  {
    /*** Initialize matrices ***/
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        matrix[i][j]= i+j;
      }
    }
  }
  apex::stop(p);
}

