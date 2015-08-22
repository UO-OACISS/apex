#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init_args(argc, argv, "apex_set_node_id unit test");
  apex_set_node_id(0);
  apex_finalize();
  apex_cleanup();
  return 0;
}

