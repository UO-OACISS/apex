#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>

#include "apex.h"
#include "apex_throttling.h"

#ifdef APEX_HAVE_RCR
#include "libenergy.h"
#endif

// this is the policy engine for APEX used to determine when contention
// is present on the socket and reduce the number of active threads

/// ----------------------------------------------------------------------------
///
/// Throttling Policy Engine
///
/// This probably wants to be ifdef under some flag -- I don't want to figure out
/// how to do this currently AKP 11/01/14
/// ----------------------------------------------------------------------------

bool apex_throttleOn = true;          // Current Throttle status
bool apex_checkThrottling = false;    // Is throttling desired
bool apex_energyThrottling = false;   // Try to save power while throttling
bool apex_final = false;              // When do we stop?

double max_watts = RCR_HIGH_POWER_LIMIT;
double min_watts = RCR_LOW_POWER_LIMIT;
int max_threads = RCR_MAX_THREADS;
int min_threads = RCR_MIN_THREADS;
int thread_cap = RCR_MAX_THREADS;
int headroom = 1; //
double moving_average = 0.0;
int window_size = 3;
int delay = 0;

inline int apex_get_thread_cap(void) {
  return thread_cap;
}

int test_pp = 0;
int apex_throttling_policy_periodic(apex_context const context) 
{
    if (apex_final) return 1; // we terminated, RCR has shut down.
    // read energy counter and memory concurrency to determine system status
    double power = apex_current_power_high();
    moving_average = ((moving_average * (window_size-1)) + power) / window_size;

    if (power != 0.0) {
      /* this is a hard limit! If we exceed the power cap once
         or in our moving average, then we need to adjust */
      if (((power > max_watts) || 
           (moving_average > max_watts)) && --delay <= 0) { 
          int amtLower = thread_cap - min_threads;
          amtLower = amtLower >> 1; // move max half way down
          if (amtLower < 0) amtLower = 1;
          thread_cap -= amtLower;
          if (thread_cap < min_threads) thread_cap = min_threads;
          //printf("power : %f, ma: %f, new cap: %d, throttling\n", power, moving_average, thread_cap);
          delay = window_size;
	  apex_throttleOn = true;
      }
      /* this is a softer limit. If we dip below the lower cap
         AND our moving average is also blow the cap, we need 
         to adjust */
      else if ((power < min_watts) && 
               (moving_average < min_watts) && --delay <=0) {
          int amtRaise = max_threads - thread_cap;
          amtRaise = amtRaise >> 1; // move max half way up
          if (amtRaise < 0) amtRaise = 1;
          thread_cap += amtRaise;
          if (thread_cap > max_threads) thread_cap = max_threads;
          //printf("power : %f, ma: %f, new cap: %d, raised %d, NOT throttling\n", power, moving_average, thread_cap, amtRaise);
          delay = window_size;
	  apex_throttleOn = false;
      //} else {
          //printf("power : %f, ma: %f, cap: %d, no change.\n", power, moving_average, thread_cap);
      }
    }
    test_pp++;
    return 1;
}


/// ----------------------------------------------------------------------------
///
/// Functions to setup and shutdown energy measurements during execution
/// Uses the APEX Policy Engine interface to dynamicly set a variable "apex_throttleOn"
/// that the scheduling code used to determine the best number of active threads
///
/// This probably wants to be ifdef under some flag -- I don't want to figure out
/// how to do this currently AKP 11/01/14
/// ----------------------------------------------------------------------------

int apex_setup_throttling()
{
  apex_set_node_id(0);
  // if desired for this execution set up throttling & final print of total energy used 
  if (getenv("HPX_THROTTLING") != NULL) {
    char * envvar = getenv("APEX_THROTTLING_MAX_THREADS");
    if (envvar != NULL) {
      max_threads = atoi(envvar);
      thread_cap = max_threads;
    }
    envvar = getenv("APEX_THROTTLING_MIN_THREADS");
    if (envvar != NULL) {
      min_threads = atoi(envvar);
    }
    envvar = getenv("APEX_THROTTLING_MAX_WATTS");
    if (envvar != NULL) {
      max_watts = atof(envvar);
    }
    envvar = getenv("APEX_THROTTLING_MIN_WATTS");
    if (envvar != NULL) {
      min_watts = atof(envvar);
    }
    apex_checkThrottling = true;
    if (getenv("HPX_ENERGY_THROTTLING") != NULL) {
      apex_energyThrottling = true;
    }
    apex_register_periodic_policy(1000000, apex_throttling_policy_periodic);
    // get an initial power reading
    apex_current_power_high();
#ifdef APEX_HAVE_RCR
    energyDaemonEnter();
#endif
  }
  else if (getenv("HPX_ENERGY") != NULL) {
    // energyDaemonInit();  // this is done in apex initialization
  }
  return(0);
}

int apex_shutdown_throttling()
{
/*
  if (apex_checkThrottling) energyDaemonTerm(); // prints energy usage
  else if (getenv("HPX_ENERGY") != NULL) {
    energyDaemonTerm();  // this is done in apex termination
  }
*/
  apex_final = true;
  printf("periodic_policy called %d times\n", test_pp);
  //apex_finalize();
  return (0);
}


