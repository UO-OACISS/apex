#include "apex.h"
#include <unistd.h>

int main (int argc, char** argv) {
  apex_init(argc, argv);
  apex_version();
  apex_set_node_id(0);
  apex_start("Main");
  sleep(2);
  apex_stop("Main");
  apex_sample_value("Apex Version", apex_version());
  apex_finalize();
  return 0;
}

