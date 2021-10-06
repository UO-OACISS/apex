#include "apex.h"
#include "stdio.h"
#include "stdlib.h"
#include <unistd.h>

/* This unit test is intended to test the APEX_DISABLE environment variable. */

int main (int argc, char** argv) {
  apex_init("apex_get_idle_rate unit test", 0, 1);
  apex_set_use_screen_output(1);
  printf("APEX Version : %s\n", apex_version());
  apex_profiler_handle main_profiler = apex_start(APEX_FUNCTION_ADDRESS,(const void*)(main));
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
  // suspend measurement
  apex_set_suspend(true);
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
  // resume measurement
  apex_set_suspend(false);
  usleep(100);
  // Call "Test Timer" 25 times
  for(i = 0; i < 25; ++i) {
    apex_profiler_handle p = apex_start(APEX_NAME_STRING,"Test Timer");
    apex_stop(p);
  }
  // The profile should show special "APEX Idle Time" counter.
  apex_profile * profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_TIME);
  if (profile) {
    printf("Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  // The profile should show special "APEX Non-Idle Time" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_NON_IDLE_TIME);
  if (profile) {
    printf("Non-Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  // The profile should show special "APEX Idle Rate" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_RATE);
  if (profile) {
    printf("Idle rate reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  apex_stop(main_profiler);
  // The profile should show special "APEX Idle Time" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_TIME);
  if (profile) {
    printf("Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  // The profile should show special "APEX Non-Idle Time" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_NON_IDLE_TIME);
  if (profile) {
    printf("Non-Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_RATE);
  if (profile) {
    printf("Idle rate reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  apex_finalize();
  // The profile should show special "APEX Idle Time" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_TIME);
  if (profile) {
    printf("Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  // The profile should show special "APEX Non-Idle Time" counter.
  profile = apex_get_profile(APEX_NAME_STRING,APEX_NON_IDLE_TIME);
  if (profile) {
    printf("Non-Idle time reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  profile = apex_get_profile(APEX_NAME_STRING,APEX_IDLE_RATE);
  if (profile) {
    printf("Idle rate reported : %f\n", profile->accumulated);
    if (profile->accumulated >= 0.0) {  // might be less, some calls might have been missed
        printf("Test passed.\n");
    }
  }
  bool passed = true;
  profile = apex_get_profile(APEX_NAME_STRING,"Test Timer");
  if (profile) {
    printf("Value Reported : %f\n", profile->calls);
    if (profile->calls > 25) {  // might be less, some calls might have been missed
        passed = false;
    }
  }
  profile = apex_get_profile(APEX_NAME_STRING,"foo");
  if (profile) {
    printf("Value Reported : %f\n", profile->calls);
    if (profile->calls > 30) {  // might be less, some calls might have been missed
        passed = false;
    }
  }
  profile = apex_get_profile(APEX_NAME_STRING,"bar");
  if (profile) {
    printf("Value Reported : %f\n", profile->calls);
    if (profile->calls > 40) {  // might be less, some calls might have been missed
        passed = false;
    }
  }
  if (passed) {
      printf("Test passed.\n");
  }
  apex_cleanup();
  return 0;
}

