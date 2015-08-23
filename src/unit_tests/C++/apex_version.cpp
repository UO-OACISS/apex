#include "apex_api.hpp"
#include <iostream>

int main (int argc, char** argv) {
  apex::init("apex::version unit test");
  std::cout << apex::version() << std::endl;
  apex::finalize();
  apex::cleanup();
  return 0;
}

