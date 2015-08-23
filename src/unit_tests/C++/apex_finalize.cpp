#include "apex_api.hpp"

int main (int argc, char** argv) {
  apex::init("apex::finalize unit test");
  apex::apex_options::use_screen_output(true);
  apex::finalize();
  apex::cleanup();
  return 0;
}

