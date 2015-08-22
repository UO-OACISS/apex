#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init_args(argc, argv, "apex_init_args unit test");
  apex_set_use_screen_output(1);
  apex_finalize();
  apex_cleanup();
  return 0;
}

