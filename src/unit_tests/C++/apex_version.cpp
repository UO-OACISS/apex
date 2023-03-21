#include "apex_api.hpp"
#include <iostream>

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex::version unit test", 0, 1);
  std::cout << apex::version() << std::endl;
  sleep(1);
  apex::finalize();
  apex::cleanup();
  return 0;
}

