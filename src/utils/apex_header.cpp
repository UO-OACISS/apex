#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  init("apex_header", 0, 1);
  finalize();
  return 0;
}

