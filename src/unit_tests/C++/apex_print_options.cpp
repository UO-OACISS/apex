#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

int main (int argc, char** argv) {
  apex_options::use_screen_output(true);
  init(argc, argv, "apex::print_options unit test");
  cout << "APEX Version : " << version() << endl;
  apex_options::print_options();
  finalize();
  return 0;
}

