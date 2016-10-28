#include "apex_api.hpp"

using namespace apex;

int main (int argc, char** argv) {
  init("apex_set_state unit test", 0, 1);
  apex_options::use_screen_output(true);
  set_state(APEX_IDLE);
  set_state(APEX_BUSY);
  set_state(APEX_THROTTLED);
  set_state(APEX_WAITING);
  set_state(APEX_BLOCKED);
  finalize();
  cleanup();
  return 0;
}

