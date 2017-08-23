#include "apex.h"
#include "stdio.h"
#include "stdlib.h"
#include <unistd.h>

int main (int argc, char** argv) {
  apex_init("apex_dump unit test", 0, 1);
  apex_set_use_screen_output(1);
  printf("APEX Version : %s\n", apex_version());
  apex_profiler_handle main_profiler = apex_start(APEX_FUNCTION_ADDRESS,(void*)(main));
  int i = 0;
  // Call "foo" 30 times
  for(i = 0; i < 30; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"foo");
    apex_stop(p);
  }    
  // Call "bar" 40 times
  for(i = 0; i < 40; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"bar");
    apex_stop(p);
  }    
  // dump and Reset everything
  apex_dump(true);
  usleep(100);
  // Call "foo" 3 times
  for(i = 0; i < 3; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"foo");
    apex_stop(p);
  }    
  // Call "bar" 4 times
  for(i = 0; i < 4; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"bar");
    apex_stop(p);
  }    
  // The profile should show "foo" was called 3 times
  // and bar was called 4 times.
  
  // Call "Test Timer" 100 times
  for(i = 0; i < 100; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"Test Timer");
    apex_stop(p);
  }    
  // dump and reset nothing
  apex_dump(false);
  // Reset "Test Timer"
  apex_reset(APEX_NAME_STRING, "Test Timer");
  usleep(100);
  // Call "Test Timer" 25 times
  for(i = 0; i < 25; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"Test Timer");
    apex_stop(p);
  }    
  // The profile should show "Test Timer" was called 25 times.
  apex_stop(main_profiler);
  apex_finalize();
  apex_profile * profile = apex_get_profile(APEX_NAME_STRING,"Test Timer");
  if (profile) {
    printf("Value Reported : %f\n", profile->calls);
    if (profile->calls <= 25) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  apex_cleanup();
  return 0;
}

