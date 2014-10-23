#include "apex.h"
#include <unistd.h>

void foo(void) {
  apex_profiler_handle profiler = apex_start_address(foo);
	sleep(1);
  apex_stop_profiler(profiler);
}

int main (int argc, char** argv) {
  apex_init_args(argc, argv, NULL);
  apex_version();
  apex_set_node_id(0);
  apex_profiler_handle profiler = apex_start_address(main);
  sleep(1);
  int i;
  for (i = 0 ; i < 3 ; i++)
    foo();
  apex_sample_value("Apex Version", apex_version());
  apex_stop_profiler(profiler);
  apex_finalize();
  return 0;
}

