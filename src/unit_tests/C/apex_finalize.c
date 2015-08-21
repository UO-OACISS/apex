#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init_args(argc, argv, "apex_finalize unit test");
  apex_finalize();
  apex_cleanup();
  return 0;
}

