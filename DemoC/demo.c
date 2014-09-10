#include "apex.h"
#include <unistd.h>

void foo(void) {
	sleep(1);
}

int main (int argc, char** argv) {
  apex_init_args(argc, argv, NULL);
  apex_version();
  apex_set_node_id(0);
  apex_start("Main");
  sleep(1);
  apex_stop("Main");
  apex_start_addr(foo);
  foo();
  apex_stop_addr(foo);
  apex_sample_value("Apex Version", apex_version());
  apex_finalize();
  return 0;
}

