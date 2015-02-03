#include "apex.hpp"
#include <unistd.h>

using namespace apex;


int main (int argc, char** argv) {
  init(argc, argv, NULL);
  version();
  set_node_id(0);
  void * profiler = start((void*)(main));
  // Call "foo" 30 times
  for(int i = 0; i < 30; ++i) {
    void * p = start("foo");
    stop(p);
  }    
  // Call "bar" 40 times
  for(int i = 0; i < 40; ++i) {
    void * p = start("bar");
    stop(p);
  }    
  // Reset everything
  reset((void *)nullptr);
  usleep(100);
  // Call "foo" 3 times
  for(int i = 0; i < 3; ++i) {
    void * p = start("foo");
    stop(p);
  }    
  // Call "bar" 4 times
  for(int i = 0; i < 4; ++i) {
    void * p = start("bar");
    stop(p);
  }    
  // The profile should show "foo" was called 3 times
  // and bar was called 4 times.
  
  // Call "Test Timer" 100 times
  for(int i = 0; i < 100; ++i) {
    void * p = start("Test Timer");
    stop(p);
  }    
  // Reset "Test Timer"
  reset("Test Timer");
  usleep(100);
  // Call "Test Timer" 25 times
  for(int i = 0; i < 25; ++i) {
    void * p = start("Test Timer");
    stop(p);
  }    
  // The profile should show "Test Timer" was called 25 times.
  stop(profiler);
  sample_value("Apex Version", version());
  finalize();
  return 0;
}

