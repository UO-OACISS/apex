#include "apex_api.hpp"

int main (int argc, char** argv) {
  apex::init("apex::init unit test", 0, 1);
  apex::apex_options::use_screen_output(true);
  apex::finalize();
  apex::cleanup();
  return 0;
}

