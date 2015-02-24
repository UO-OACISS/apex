#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "apex.h"
#include "apex_types.h"
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

int test_pp = 0;

#define MAX_WINDOW_SIZE 3
// variables related to power throttling
double max_watts = RCR_HIGH_POWER_LIMIT;
double min_watts = RCR_LOW_POWER_LIMIT;
int max_threads = RCR_MAX_THREADS;
int min_threads = RCR_MIN_THREADS;
int thread_cap = RCR_MAX_THREADS;
int headroom = 1; //
double moving_average = 0.0;
int window_size = MAX_WINDOW_SIZE;
int delay = 0;

// variables related to throughput throttling
apex_function_address function_of_interest = APEX_NULL_FUNCTION_ADDRESS;
char * function_name_of_interest = NULL;
apex_profile function_baseline;
apex_profile function_history;
int throughput_delay = MAX_WINDOW_SIZE; // initialize 
typedef enum {INITIAL_STATE, BASELINE, INCREASE, DECREASE, NO_CHANGE} last_action_t;
last_action_t last_action = INITIAL_STATE;
apex_optimization_criteria_t throttling_criteria = MAXIMIZE_THROUGHPUT;

inline int apex_get_thread_cap(void) {
  return thread_cap;
}

inline void decrease_cap() {
    int amtLower = thread_cap - min_threads;
    amtLower = amtLower >> 1; // move max half way down
    if (amtLower <= 0) amtLower = 1;
    thread_cap -= amtLower;
    if (thread_cap < min_threads) thread_cap = min_threads;
    //printf("%d more throttling! new cap: %d\n", test_pp, thread_cap); fflush(stdout);
    apex_throttleOn = true;
}

inline void increase_cap() {
    int amtRaise = max_threads - thread_cap;
    amtRaise = amtRaise >> 1; // move max half way up
    if (amtRaise <= 0) {
        amtRaise = 1;
    }
    thread_cap += amtRaise;
    if (thread_cap > max_threads) {
        thread_cap = max_threads;
    }
    //printf("%d less throttling! new cap: %d\n", test_pp, thread_cap); fflush(stdout);
    apex_throttleOn = false;
}

inline int apex_power_throttling_policy(apex_context const context) 
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
          decrease_cap();
          delay = window_size;
      }
      /* this is a softer limit. If we dip below the lower cap
         AND our moving average is also blow the cap, we need 
         to adjust */
      else if ((power < min_watts) && 
               (moving_average < min_watts) && --delay <=0) {
          increase_cap();
          delay = window_size;
      //} else {
          //printf("power : %f, ma: %f, cap: %d, no change.\n", power, moving_average, thread_cap);
      }
    }
    test_pp++;
    return 1;
}

int apex_throughput_throttling_policy(apex_context const context) {
// Do we have a function of interest?
//    No: do nothing, return.
//    Yes: Get its profile, continue.
    test_pp++;
    if(function_of_interest == APEX_NULL_FUNCTION_ADDRESS &&
       function_name_of_interest == NULL) { 
        //printf("%d No function.\n", test_pp);
        return 1; 
    }

    throughput_delay--;
    
// Do we have sufficient history for this function?
//    No: save these results, wait for more data (3 iterations, at least), return
//    Yes: get the profile

    if(throughput_delay > 0) { 
      //printf("%d Waiting...\n", test_pp);
      return 1; 
    }

    if (throughput_delay == 0) {
      // reset the profile for clean measurement
      //printf("%d Taking measurement...\n", test_pp);
      if(function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
          apex_reset_address(function_of_interest); // we want new measurements!
      } else {
          apex_reset_name(function_name_of_interest); // we want new measurements!
      }
      return 1;
    }

    apex_profile * function_profile = NULL;
    if(function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
        function_profile = apex_get_profile_from_address(function_of_interest);
    } else {
        function_profile = apex_get_profile_from_name(function_name_of_interest);
    }
    double current_mean = function_profile->accumulated / function_profile->calls;
    //printf("%d Calls: %f, Accum: %f, Mean: %f\n", test_pp, function_profile->calls, function_profile->accumulated, current_mean);

// first time, take a baseline measurement 
    if (last_action == INITIAL_STATE) {
        function_baseline.calls = function_profile->calls;
        function_baseline.accumulated = function_profile->accumulated;
        function_history.calls = function_profile->calls;
        function_history.accumulated = function_profile->accumulated;
        throughput_delay = MAX_WINDOW_SIZE;
        last_action = BASELINE;
        //printf("%d Got baseline.\n", test_pp);
    }

    //printf("%d Old: %f New %f.\n", test_pp, function_history.calls, function_profile->calls);

//    Subsequent times: Are we doing better than before?
//       No: compare profile to history, adjust as necessary.
//       Yes: compare profile to history, adjust as necessary.
    bool do_decrease = false;
    bool do_increase = false;

    // first time, try decreasing the number of threads.
    if (last_action == BASELINE) {
        do_decrease = true;
    } else if (throttling_criteria == MAXIMIZE_THROUGHPUT) {
        // are we at least 5% more throughput? If so, do more adjustment
        if (function_profile->calls > (1.05*function_history.calls)) {
            if (last_action == INCREASE) { do_increase = true; }
            else if (last_action == DECREASE) { do_decrease = true; }
        // are we at least 5% less throughput? If so, reverse course
        } else if (function_profile->calls < (0.95*function_history.calls)) {
            if (last_action == DECREASE) { do_increase = true; }
            else if (last_action == INCREASE) { do_decrease = true; }
        } else {
            double old_mean = function_history.accumulated / function_history.calls;
            // are we at least 5% more efficient? If so, do more adjustment
            if (old_mean > (1.05*current_mean)) {
                if (last_action == INCREASE) { do_increase = true; }
                else if (last_action == DECREASE) { do_decrease = true; }
            // are we at least 5% less efficient? If so, reverse course
            } else if (old_mean < (0.95*current_mean)) {
                if (last_action == DECREASE) { do_increase = true; }
                else if (last_action == INCREASE) { do_decrease = true; }
            } else {
            // otherwise, nothing to do.
            }
        }
    } else if (throttling_criteria == MAXIMIZE_ACCUMULATED) {
        double old_mean = function_history.accumulated / function_history.calls;
        // are we at least 5% more efficient? If so, do more adjustment
        if (old_mean > (1.05*current_mean)) {
            if (last_action == INCREASE) { do_increase = true; }
            else if (last_action == DECREASE) { do_decrease = true; }
        // are we at least 5% less efficient? If so, reverse course
        } else if (old_mean < (0.95*current_mean)) {
            if (last_action == DECREASE) { do_increase = true; }
            else if (last_action == INCREASE) { do_decrease = true; }
        } else {
        // otherwise, nothing to do.
        }
    } else if (throttling_criteria == MINIMIZE_ACCUMULATED) {
        double old_mean = function_history.accumulated / function_history.calls;
        // are we at least 5% less efficient? If so, reverse course
        if (old_mean > (1.05*current_mean)) {
            if (last_action == DECREASE) { do_increase = true; }
            else if (last_action == INCREASE) { do_decrease = true; }
        // are we at least 5% more efficient? If so, do more adjustment
        } else if (old_mean < (0.95*current_mean)) {
            if (last_action == INCREASE) { do_increase = true; }
            else if (last_action == DECREASE) { do_decrease = true; }
        } else {
        // otherwise, nothing to do.
        }
    }

    if (do_decrease) {
        //printf("%d Decreasing.\n", test_pp);
        // save this as our new history
        function_history.calls = function_profile->calls;
        function_history.accumulated = function_profile->accumulated;
        decrease_cap();
        last_action = DECREASE;
    } else if (do_increase) {
        //printf("%d Increasing.\n", test_pp);
        // save this as our new history
        function_history.calls = function_profile->calls;
        function_history.accumulated = function_profile->accumulated;
        increase_cap();
        last_action = INCREASE;
    }
    throughput_delay = MAX_WINDOW_SIZE;
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

int apex_setup_power_cap_throttling()
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
      apex_register_periodic_policy(1000000, apex_power_throttling_policy);
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

APEX_EXPORT int apex_setup_address_throttling(apex_function_address the_address,
                apex_optimization_criteria_t criteria)
{
    function_of_interest = the_address;
    function_history.calls = 0.0;
    function_history.accumulated = 0.0;
    function_baseline.calls = 0.0;
    function_baseline.accumulated = 0.0;
    throttling_criteria = criteria;
    apex_register_periodic_policy(1000000, apex_throughput_throttling_policy);
    return(0);
}

APEX_EXPORT int apex_setup_name_throttling(const char * the_name,
                apex_optimization_criteria_t criteria)
{
    if (the_name == NULL) {
        fprintf(stderr, "Timer/counter name for throttling is null. Please specify a name.\n");
        abort();
    }
    function_name_of_interest = malloc(sizeof(char) * strlen(the_name));
    strcpy (function_name_of_interest, the_name);
    function_history.calls = 0.0;
    function_history.accumulated = 0.0;
    function_baseline.calls = 0.0;
    function_baseline.accumulated = 0.0;
    throttling_criteria = criteria;
    apex_register_periodic_policy(1000000, apex_throughput_throttling_policy);
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
  //printf("periodic_policy called %d times\n", test_pp);
  //apex_finalize();
  return (0);
}


