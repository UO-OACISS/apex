#include "matmult_initialize.h"
#include "apex.hpp"

void initialize(double **matrix, int rows, int cols) {
  apex::start("initialize()");
  int i,j;
  {
    /*** Initialize matrices ***/
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        matrix[i][j]= i+j;
      }
    }
  }
  apex::stop("initialize()");
}

