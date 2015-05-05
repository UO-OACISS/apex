#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int foo(int i) {
  apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &foo);
  int j = i * i;
  apex_stop(profiler);
  return j;
}

int main (int argc, char** argv) {
  apex_init_args(argc, argv, NULL);
  printf("APEX Version : %s\n", apex_version());
  apex_set_node_id(0);
  apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
  int i,j = 0;
  for (i = 0 ; i < 3 ; i++)
    j += foo(i);
  apex_stop(profiler);
  apex_stop_measurement();
  apex_finalize();
  return 0;
}

