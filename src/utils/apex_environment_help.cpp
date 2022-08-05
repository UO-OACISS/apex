#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  init("apex_environment_help", 0, 1);
  cout << "APEX Version : " << version() << endl;
  apex_options::environment_help();
  finalize();
  return 0;
}

