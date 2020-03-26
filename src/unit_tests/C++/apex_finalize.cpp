#include "apex_api.hpp"

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex::finalize unit test", 0, 1);
  apex::apex_options::use_screen_output(true);
  apex::finalize();
  apex::cleanup();
  return 0;
}

