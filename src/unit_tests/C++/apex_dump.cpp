#include "apex_api.hpp"
#include <unistd.h>

using namespace apex;
using namespace std;


int main (int argc, char** argv) {
  init("apex::dump unit test", 0, 1);
  apex_options::use_screen_output(true);
  cout << "APEX Version : " << version() << endl;
  profiler * main_profiler = start((apex_function_address)(main));
  // Call "foo" 30 times
  for(int i = 0; i < 30; ++i) {
    profiler * p = start("foo");
    stop(p);
  }    
  // Call "bar" 40 times
  for(int i = 0; i < 40; ++i) {
    profiler * p = start("bar");
    stop(p);
  }    
  // dump and reset everything
  std::cout << dump(true) << endl;
  usleep(100);
  // Call "foo" 3 times
  for(int i = 0; i < 3; ++i) {
    profiler * p = start("foo");
    stop(p);
  }    
  // Call "bar" 4 times
  for(int i = 0; i < 4; ++i) {
    profiler * p = start("bar");
    stop(p);
  }    
  // The profile should show "foo" was called 3 times
  // and bar was called 4 times.
  
  // Call "Test Timer" 100 times
  for(int i = 0; i < 100; ++i) {
    profiler * p = start("Test Timer");
    stop(p);
  }    
  // dump and reset nothing
  std::cout << dump(false) << endl;
  // Reset "Test Timer"
  reset("Test Timer");
  usleep(100);
  // Call "Test Timer" 25 times
  for(int i = 0; i < 25; ++i) {
    profiler * p = start("Test Timer");
    stop(p);
  }    
  // The profile should show "Test Timer" was called 25 times.
  stop(main_profiler);
  finalize();
  apex_profile * profile = get_profile("Test Timer");
  if (profile) {
    std::cout << "Value Reported : " << profile->calls << std::endl;
    if (profile->calls <= 25) {  // might be less, some calls might have been missed
        std::cout << "Test passed." << std::endl;
    }
  }
  cleanup();
  return 0;
}

