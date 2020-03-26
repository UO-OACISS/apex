#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  init("apex_make_default_config", 0, 1);
  cout << "APEX Version : " << version() << endl;
  apex_options::make_default_config();
  finalize();
  return 0;
}

