#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init("apex_version unit test", 0, 1);
  printf("APEX Version : %s\n", apex_version());
  apex_finalize();
  apex_cleanup();
  return 0;
}

