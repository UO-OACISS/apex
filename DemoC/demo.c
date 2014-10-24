#include "apex.h"
#include <unistd.h>

int foo(int i) {
  apex_profiler_handle profiler = apex_start_address(foo);
  int j = i * i;
  apex_stop_profiler(profiler);
  return j;
}

int main (int argc, char** argv) {
  apex_init_args(argc, argv, NULL);
  apex_version();
  apex_set_node_id(0);
  apex_profiler_handle profiler = apex_start_address(main);
  //sleep(1);
  int i,j = 0;
  for (i = 0 ; i < 3 ; i++)
    j += foo(i);
  apex_sample_value("Apex Version", apex_version());
  apex_stop_profiler(profiler);
  apex_finalize();
  return 0;
}

