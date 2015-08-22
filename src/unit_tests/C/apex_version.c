#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init_args(argc, argv, "apex_version unit test");
  printf("APEX Version : %s\n", apex_version());
  apex_finalize();
  apex_cleanup();
  return 0;
}

