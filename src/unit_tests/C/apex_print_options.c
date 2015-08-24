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
  apex_set_use_screen_output(1);
  apex_init_args(argc, argv, "apex_print_options unit test");
  printf("APEX Version : %s\n", apex_version());
  apex_print_options();
  apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
  int i,j = 0;
  for (i = 0 ; i < 3 ; i++)
    j += foo(i);
  apex_stop(profiler);
  apex_finalize();
  apex_cleanup();
  return 0;
}

