#include "apex.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char** argv) {
  apex_init("apex_set_state unit test");
  apex_profiler_handle p = apex_start(APEX_NAME_STRING, "main");
  apex_set_use_screen_output(1);
  apex_set_state(APEX_IDLE);
  apex_set_state(APEX_BUSY);
  apex_set_state(APEX_THROTTLED);
  apex_set_state(APEX_WAITING);
  apex_set_state(APEX_BLOCKED);
  apex_stop(p);
  apex_finalize();
  apex_cleanup();
  return 0;
}

