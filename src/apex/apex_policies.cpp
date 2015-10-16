#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include "apex.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <boost/atomic.hpp>

#include "apex.hpp"
#include "apex_api.hpp"
#include "apex_types.h"
#include "apex_policies.hpp"
#include "apex_options.hpp"
#include "utils.hpp"

#ifdef APEX_HAVE_RCR
#include "libenergy.h"
#endif

#ifdef APEX_HAVE_ACTIVEHARMONY
#include "hclient.h"
#endif

using namespace std;

// this is the policy engine for APEX used to determine when contention
// is present on the socket and reduce the number of active threads

bool apex_throttleOn = true;          // Current Throttle status
bool apex_checkThrottling = false;    // Is throttling desired
bool apex_energyThrottling = false;   // Try to save power while throttling
bool apex_final = false;              // When do we stop?

static apex_tuning_session global_ts;
static apex_tuning_session * tuning_session = &global_ts;



inline int __get_thread_cap(void) {
  return (int)(tuning_session->thread_cap);
  //return (int)*(tuning_session->__ah_inputs[0]);
}

inline void __set_thread_cap(int new_cap) {
  tuning_session->thread_cap = (long int)new_cap;
  return;
}

#if 0  // unused for now
inline int __get_inputs(long int **inputs, int * num_inputs) {
  inputs = &(tuning_session->__ah_inputs[0]);
  *num_inputs = tuning_session->__num_ah_inputs;
  return tuning_session->__num_ah_inputs;
}
#endif

inline void __decrease_cap_gradual() {
    tuning_session->thread_cap -= 1;
    if (tuning_session->thread_cap < tuning_session->min_threads) { tuning_session->thread_cap = tuning_session->min_threads; }
#ifdef APEX_DEBUG_THROTTLE
    printf("%d more throttling! new cap: %d\n", tuning_session->test_pp, tuning_session->thread_cap); fflush(stdout);
#endif
    //apex_throttleOn = true;
}

inline void __decrease_cap() {
    int amtLower = tuning_session->thread_cap - tuning_session->min_threads;
    amtLower = amtLower >> 1; // move max half way down
    if (amtLower <= 0) amtLower = 1;
    tuning_session->thread_cap -= amtLower;
    if (tuning_session->thread_cap < tuning_session->min_threads) tuning_session->thread_cap = tuning_session->min_threads;
#ifdef APEX_DEBUG_THROTTLE
    printf("%d more throttling! new cap: %d\n", tuning_session->test_pp, tuning_session->thread_cap); fflush(stdout);
#endif
    //apex_throttleOn = true;
}

inline void __increase_cap_gradual() {
    tuning_session->thread_cap += 1;
    if (tuning_session->thread_cap > tuning_session->max_threads) { tuning_session->thread_cap = tuning_session->max_threads; }
#ifdef APEX_DEBUG_THROTTLE
    printf("%d less throttling! new cap: %d\n", tuning_session->test_pp, tuning_session->thread_cap); fflush(stdout);
#endif
    //apex_throttleOn = false;
}

inline void __increase_cap() {
    int amtRaise = tuning_session->max_threads - tuning_session->thread_cap;
    amtRaise = amtRaise >> 1; // move max half way up
    if (amtRaise <= 0) {
        amtRaise = 1;
    }
    tuning_session->thread_cap += amtRaise;
    if (tuning_session->thread_cap > tuning_session->max_threads) {
        tuning_session->thread_cap = tuning_session->max_threads;
    }
#ifdef APEX_DEBUG_THROTTLE
    printf("%d less throttling! new cap: %d\n", tuning_session->test_pp, tuning_session->thread_cap); fflush(stdout);
#endif
    //apex_throttleOn = false;
}

inline int apex_power_throttling_policy(apex_context const context) 
{
    APEX_UNUSED(context);
    if (apex_final) return APEX_NOERROR; // we terminated, RCR has shut down.
    //if (apex::apex::instance()->get_node_id() == 0) return APEX_NOERROR; 
    // read energy counter and memory concurrency to determine system status
    double power = apex::current_power_high();
    tuning_session->moving_average = ((tuning_session->moving_average * (tuning_session->window_size-1)) + power) / tuning_session->window_size;
    //cout << "power in policy is: " << power << endl;

    if (power != 0.0) {
      /* this is a hard limit! If we exceed the power cap once
         or in our moving average, then we need to adjust */
      --tuning_session->delay;
      if (((power > tuning_session->max_watts) ||
           (tuning_session->moving_average > tuning_session->max_watts)) && tuning_session->delay <= 0) { 
          __decrease_cap();
          tuning_session->delay = tuning_session->window_size;
      }
      /* this is a softer limit. If we dip below the lower cap
         AND our moving average is also blow the cap, we need 
         to adjust */
      else if ((power < tuning_session->min_watts) && 
               (tuning_session->moving_average < tuning_session->min_watts) && tuning_session->delay <=0) {
          __increase_cap_gradual();
          tuning_session->delay = tuning_session->window_size;
      } else {
#ifdef APEX_DEBUG_THROTTLE
          printf("power : %f, ma: %f, cap: %d, min: %f, max: %f, tuning_session->delay: %d no change.\n", power, tuning_session->moving_average, tuning_session->thread_cap, tuning_session->min_watts, tuning_session->max_watts, tuning_session->delay);
#endif
      }
      apex::apex * instance = apex::apex::instance();
      if (instance != NULL && instance->get_node_id() == 0) {
        static int index = 0;
        tuning_session->cap_data << index++ << "\t" << power << "\t" << tuning_session->thread_cap << endl;
      }
    }
    tuning_session->test_pp++;
    return APEX_NOERROR;
}

int apex_throughput_throttling_policy(apex_context const context) {
    APEX_UNUSED(context);
// Do we have a function of interest?
//    No: do nothing, return.
//    Yes: Get its profile, continue.
    tuning_session->test_pp++;
    if(tuning_session->function_of_interest == APEX_NULL_FUNCTION_ADDRESS &&
       tuning_session->function_name_of_interest == "") { 
        //printf("%d No function.\n", tuning_session->test_pp);
        return APEX_NOERROR; 
    }

    tuning_session->throughput_delay--;
    
// Do we have sufficient history for this function?
//    No: save these results, wait for more data (3 iterations, at least), return
//    Yes: get the profile

    if(tuning_session->throughput_delay > 0) { 
      //printf("%d Waiting...\n", tuning_session->test_pp);
      return APEX_NOERROR; 
    }

    if (tuning_session->throughput_delay == 0) {
      // reset the profile for clean measurement
      //printf("%d Taking measurement...\n", tuning_session->test_pp);
      if(tuning_session->function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
          apex::reset(tuning_session->function_of_interest); // we want new measurements!
      } else {
          apex::reset(tuning_session->function_name_of_interest); // we want new measurements!
      }
      return APEX_NOERROR;
    }

    apex_profile * function_profile = NULL;
    if(tuning_session->function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
        function_profile = apex::get_profile(tuning_session->function_of_interest);
    } else {
        function_profile = apex::get_profile(tuning_session->function_name_of_interest);
    }
    double current_mean = function_profile->accumulated / function_profile->calls;
    //printf("%d Calls: %f, Accum: %f, Mean: %f\n", tuning_session->test_pp, function_profile->calls, function_profile->accumulated, current_mean);

// first time, take a baseline measurement 
    if (tuning_session->last_action == INITIAL_STATE) {
        tuning_session->function_baseline.calls = function_profile->calls;
        tuning_session->function_baseline.accumulated = function_profile->accumulated;
        tuning_session->function_history.calls = function_profile->calls;
        tuning_session->function_history.accumulated = function_profile->accumulated;
        tuning_session->throughput_delay = MAX_WINDOW_SIZE;
        tuning_session->last_action = BASELINE;
        //printf("%d Got baseline.\n", tuning_session->test_pp);
    }

    //printf("%d Old: %f New %f.\n", tuning_session->test_pp, tuning_session->function_history.calls, function_profile->calls);

//    Subsequent times: Are we doing better than before?
//       No: compare profile to history, adjust as necessary.
//       Yes: compare profile to history, adjust as necessary.
    bool do_decrease = false;
    bool do_increase = false;

    // first time, try decreasing the number of threads.
    if (tuning_session->last_action == BASELINE) {
        do_decrease = true;
    } else if (tuning_session->throttling_criteria == APEX_MAXIMIZE_THROUGHPUT) {
        // are we at least 5% more throughput? If so, do more adjustment
        if (function_profile->calls > (1.05*tuning_session->function_history.calls)) {
            if (tuning_session->last_action == INCREASE) { do_increase = true; }
            else if (tuning_session->last_action == DECREASE) { do_decrease = true; }
        // are we at least 5% less throughput? If so, reverse course
        } else if (function_profile->calls < (0.95*tuning_session->function_history.calls)) {
            if (tuning_session->last_action == DECREASE) { do_increase = true; }
            else if (tuning_session->last_action == INCREASE) { do_decrease = true; }
        } else {
#if 1
            double old_mean = tuning_session->function_history.accumulated / tuning_session->function_history.calls;
            // are we at least 5% more efficient? If so, do more adjustment
            if (old_mean > (1.05*current_mean)) {
                if (tuning_session->last_action == INCREASE) { do_increase = true; }
                else if (tuning_session->last_action == DECREASE) { do_decrease = true; }
            // are we at least 5% less efficient? If so, reverse course
            } else if (old_mean < (0.95*current_mean)) {
                if (tuning_session->last_action == DECREASE) { do_increase = true; }
                else if (tuning_session->last_action == INCREASE) { do_decrease = true; }
            } else {
            // otherwise, nothing to do.
            }
#endif
        }
    } else if (tuning_session->throttling_criteria == APEX_MAXIMIZE_ACCUMULATED) {
        double old_mean = tuning_session->function_history.accumulated / tuning_session->function_history.calls;
        // are we at least 5% more efficient? If so, do more adjustment
        if (old_mean > (1.05*current_mean)) {
            if (tuning_session->last_action == INCREASE) { do_increase = true; }
            else if (tuning_session->last_action == DECREASE) { do_decrease = true; }
        // are we at least 5% less efficient? If so, reverse course
        } else if (old_mean < (0.95*current_mean)) {
            if (tuning_session->last_action == DECREASE) { do_increase = true; }
            else if (tuning_session->last_action == INCREASE) { do_decrease = true; }
        } else {
        // otherwise, nothing to do.
        }
    } else if (tuning_session->throttling_criteria == APEX_MINIMIZE_ACCUMULATED) {
        double old_mean = tuning_session->function_history.accumulated / tuning_session->function_history.calls;
        // are we at least 5% less efficient? If so, reverse course
        if (old_mean > (1.05*current_mean)) {
            if (tuning_session->last_action == DECREASE) { do_increase = true; }
            else if (tuning_session->last_action == INCREASE) { do_decrease = true; }
        // are we at least 5% more efficient? If so, do more adjustment
        } else if (old_mean < (0.95*current_mean)) {
            if (tuning_session->last_action == INCREASE) { do_increase = true; }
            else if (tuning_session->last_action == DECREASE) { do_decrease = true; }
        } else {
        // otherwise, nothing to do.
        }
    }

    if (do_decrease) {
        //printf("%d Decreasing.\n", tuning_session->test_pp);
        // save this as our new history
        tuning_session->function_history.calls = function_profile->calls;
        tuning_session->function_history.accumulated = function_profile->accumulated;
        __decrease_cap_gradual();
        tuning_session->last_action = DECREASE;
    } else if (do_increase) {
        //printf("%d Increasing.\n", tuning_session->test_pp);
        // save this as our new history
        tuning_session->function_history.calls = function_profile->calls;
        tuning_session->function_history.accumulated = function_profile->accumulated;
        __increase_cap_gradual();
        tuning_session->last_action = INCREASE;
    }
    tuning_session->throughput_delay = MAX_WINDOW_SIZE;
    return APEX_NOERROR;
}

/* How about a hill-climbing method for throughput? */

/* Discrete Space Hill Climbing Algorithm */
int apex_throughput_throttling_dhc_policy(apex_context const context) {
    APEX_UNUSED(context);

#ifdef APEX_DEBUG_THROTTLE
    printf("Throttling on name: %s\n", tuning_session->function_name_of_interest.c_str());
#endif

    // initial value for current_cap is 1/2 the distance between min and max
    static double previous_value = 0.0; // instead of resetting.
    //static int current_cap = tuning_session->min_threads + ((tuning_session->max_threads - tuning_session->min_threads) >> 1);
    static int current_cap = tuning_session->max_threads - 2;
    int low_neighbor = max(current_cap - 2, tuning_session->min_threads);
    int high_neighbor = min(current_cap + 2, tuning_session->max_threads);
    static bool got_center = false;
    static bool got_low = false;
    static bool got_high = false;

    apex_profile * function_profile = NULL;
    // get a measurement of our current setting
    if(tuning_session->function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
        function_profile = apex::get_profile(tuning_session->function_of_interest);
        //reset(tuning_session->function_of_interest); // we want new measurements!
    } else {
        function_profile = apex::get_profile(tuning_session->function_name_of_interest);
        //reset(tuning_session->function_name_of_interest); // we want new measurements!
    }
    // if we have no data yet, return.
    if (function_profile == NULL) { 
        printf ("No Data?\n");
#if defined(APEX_HAVE_HPX3) && defined(APEX_DEBUG_THROTTLE)
        std::vector<std::string> available = apex::get_available_profiles();
        for(std::string s : available) {
            printf("\"%s\"\n", s.c_str());
        }
#endif
        return APEX_ERROR; 
    } else {
#ifdef APEX_DEBUG_THROTTLE
        printf ("Got Data!\n");
#endif
    }

    double new_value = 0.0;
    if (tuning_session->throttling_criteria == APEX_MAXIMIZE_THROUGHPUT) {
        new_value = function_profile->calls - previous_value;
        previous_value = function_profile->calls;
    } else {
        new_value = function_profile->accumulated - previous_value;
        previous_value = function_profile->accumulated;
    }

    // update the moving average
    if ((++tuning_session->observations[tuning_session->thread_cap]) < tuning_session->window_size) {
        tuning_session->evaluations[tuning_session->thread_cap] = ((tuning_session->evaluations[tuning_session->thread_cap] * (tuning_session->observations[tuning_session->thread_cap]-1)) + new_value) / tuning_session->observations[tuning_session->thread_cap];
    } else {
        tuning_session->evaluations[tuning_session->thread_cap] = ((tuning_session->evaluations[tuning_session->thread_cap] * (tuning_session->window_size-1)) + new_value) / tuning_session->window_size;
    }
#ifdef APEX_DEBUG_THROTTLE
    printf("%d Value: %f, new average: %f.\n", tuning_session->thread_cap, new_value, tuning_session->evaluations[tuning_session->thread_cap]);
#endif

    if (tuning_session->thread_cap == current_cap) got_center = true;
    if (tuning_session->thread_cap == low_neighbor) got_low = true;
    if (tuning_session->thread_cap == high_neighbor) got_high = true;

    // check if our center has a value
    if (!got_center) {
        tuning_session->thread_cap = current_cap;
#ifdef APEX_DEBUG_THROTTLE
        printf("initial throttling. trying cap: %d\n", tuning_session->thread_cap); fflush(stdout);
#endif
        return APEX_NOERROR;
    }
    // check if our left of center has a value
    if (!got_low) {
        tuning_session->thread_cap = low_neighbor;
#ifdef APEX_DEBUG_THROTTLE
        printf("current-1 throttling. trying cap: %d\n", tuning_session->thread_cap); fflush(stdout);
#endif
        return APEX_NOERROR;
    }
    // check if our right of center has a value
    if (!got_high) {
        tuning_session->thread_cap = high_neighbor;
#ifdef APEX_DEBUG_THROTTLE
        printf("current+1 throttling. trying cap: %d\n", tuning_session->thread_cap); fflush(stdout);
#endif
        return APEX_NOERROR;
    }

    // clear our non-best tuning_session->observations, and set a new cap.
    int best = current_cap;

    if ((tuning_session->throttling_criteria == APEX_MAXIMIZE_THROUGHPUT) ||
        (tuning_session->throttling_criteria == APEX_MAXIMIZE_ACCUMULATED)) {
        if (tuning_session->evaluations[low_neighbor] > tuning_session->evaluations[current_cap]) {
            best = low_neighbor;
        }
        if (tuning_session->evaluations[high_neighbor] > tuning_session->evaluations[best]) {
            best = high_neighbor;
        }
    } else {
        if (tuning_session->evaluations[low_neighbor] < tuning_session->evaluations[current_cap]) {
            best = low_neighbor;
        }
        if (tuning_session->evaluations[high_neighbor] < tuning_session->evaluations[best]) {
            best = high_neighbor;
        }
    }
#ifdef APEX_DEBUG_THROTTLE
    printf("%d Calls: %f.\n", tuning_session->thread_cap, tuning_session->evaluations[best]);
    printf("New cap: %d\n", best); fflush(stdout);
#endif
    apex::apex * instance = apex::apex::instance();
    if (instance == NULL) return APEX_NOERROR;
    if (instance->get_node_id() == 0) {
        static int index = 0;
        tuning_session->cap_data << index++ << "\t" << tuning_session->evaluations[best] << "\t" << best << endl;
    }
    // set a new cap
    tuning_session->thread_cap = current_cap = best;
    got_center = false;
    got_low = false;
    got_high = false;
    return APEX_NOERROR;
}
    
#ifdef APEX_HAVE_ACTIVEHARMONY
int apex_throughput_throttling_ah_policy(apex_context const context) {
    // do something.
    APEX_UNUSED(context);
    static double previous_value = 0.0; // instead of resetting.
    static bool _converged_message = false;
    if (harmony_converged(tuning_session->hdesc)) {
        if (!_converged_message) {
            _converged_message = true;
            cout << "Thread Cap value optimization has converged." << endl;
            cout << "Thread Cap value : " << tuning_session->thread_cap << endl;
        }
        //return APEX_NOERROR;
    }

    // get a measurement of our current setting
    apex_profile * function_profile = NULL;
    if(tuning_session->function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
        function_profile = apex::get_profile(tuning_session->function_of_interest);
        //reset(tuning_session->function_of_interest); // we want new measurements!
    } else {
        function_profile = apex::get_profile(tuning_session->function_name_of_interest);
        //reset(tuning_session->function_name_of_interest); // we want new measurements!
    }
    // if we have no data yet, return.
    if (function_profile == NULL) { 
        cerr << "No profile data?" << endl;
        return APEX_ERROR; 
    //} else {
        //printf ("Got Data!\n");
    }

    double new_value = 0.0;
    if (tuning_session->throttling_criteria == APEX_MAXIMIZE_THROUGHPUT) {
        new_value = (function_profile->calls - previous_value) * -1.0;
        previous_value = function_profile->calls;
    } else if (tuning_session->throttling_criteria == APEX_MAXIMIZE_ACCUMULATED) {
        new_value = (function_profile->accumulated - previous_value) * -1.0;
        previous_value = function_profile->accumulated;
    } else if (tuning_session->throttling_criteria == APEX_MINIMIZE_ACCUMULATED) {
        new_value = function_profile->accumulated - previous_value;
        previous_value = function_profile->accumulated;
    }
    cout << "Cap: " << tuning_session->thread_cap << " New: " << abs(new_value) << " Prev: " << previous_value << endl;

    apex::apex * instance = apex::apex::instance();
    if (instance == NULL) return APEX_NOERROR;
    if (instance->get_node_id() == 0) {
        static int index = 0;
        tuning_session->cap_data << index++ << "\t" << abs(new_value) << "\t" << tuning_session->thread_cap << endl;
    }

    /* Report the performance we've just measured. */
    if (harmony_report(tuning_session->hdesc, new_value) != 0) {
        cerr << "Failed to report performance to server." << endl;
        return APEX_ERROR;
    }

    int hresult = harmony_fetch(tuning_session->hdesc);
    if (hresult < 0) {
        cerr << "Failed to fetch values from server: " << 
                harmony_error_string(tuning_session->hdesc) << endl;
        return APEX_ERROR;
    }
    else if (hresult == 0) {
        /* New values were not available at this time.
         * Bundles remain unchanged by Harmony system.
         */
    }
    else if (hresult > 0) {
        /* The Harmony system modified the variable values.
         * Do any systemic updates to deal with such a change.
         */
    }

    return APEX_NOERROR;
}

int apex_throughput_tuning_policy(apex_context const context) {
    // do something.
    APEX_UNUSED(context);
    static double previous_value = 0.0; // instead of resetting.
    static bool _converged_message = false;
    if (harmony_converged(tuning_session->hdesc)) {
        if (!_converged_message) {
            _converged_message = true;
            cout << "Tuning has converged." << endl;
        }
        return APEX_NOERROR;
    }

    // get a measurement of our current setting
    apex_profile * function_profile = NULL;
    if(tuning_session->function_of_interest != APEX_NULL_FUNCTION_ADDRESS) {
        function_profile = apex::get_profile(tuning_session->function_of_interest);
    } else {
        function_profile = apex::get_profile(tuning_session->function_name_of_interest);
    }
    // if we have no data yet, return.
    if (function_profile == NULL) { 
        cerr << "No profile data?" << endl;
        return APEX_ERROR; 
    }

    double new_value = 0.0;
    if (tuning_session->throttling_criteria == APEX_MAXIMIZE_THROUGHPUT) {
        new_value = (function_profile->calls - previous_value) * -1.0;
        previous_value = function_profile->calls;
    } else if (tuning_session->throttling_criteria == APEX_MAXIMIZE_ACCUMULATED) {
        new_value = (function_profile->accumulated - previous_value) * -1.0;
        previous_value = function_profile->accumulated;
    } else if (tuning_session->throttling_criteria == APEX_MINIMIZE_ACCUMULATED) {
        new_value = function_profile->accumulated - previous_value;
        previous_value = function_profile->accumulated;
    }

    /* Report the performance we've just measured. */
    if (harmony_report(tuning_session->hdesc, new_value) != 0) {
        cerr << "Failed to report performance to server." << endl;
        return APEX_ERROR;
    }

    int hresult = harmony_fetch(tuning_session->hdesc);
    if (hresult < 0) {
        cerr << "Failed to fetch values from server: " << 
                harmony_error_string(tuning_session->hdesc) << endl;
        return APEX_ERROR;
    }
    else if (hresult == 0) {
        /* New values were not available at this time.
         * Bundles remain unchanged by Harmony system.
         */
    }
    else if (hresult > 0) {
        /* The Harmony system modified the variable values.
         * Do any systemic updates to deal with such a change.
         */
    }

    return APEX_NOERROR;
}


int apex_custom_tuning_policy(apex_context const context) {
    APEX_UNUSED(context);
    static bool _converged_message = false;
    if (harmony_converged(tuning_session->hdesc)) {
        if (!_converged_message) {
            _converged_message = true;
            cout << "Tuning has converged." << endl;
        }
        return APEX_NOERROR;
    }

    // get a measurement of our current setting
    double new_value = tuning_session->metric_of_interest();

    /* Report the performance we've just measured. */
    if (harmony_report(tuning_session->hdesc, new_value) != 0) {
        cerr << "Failed to report performance to server." << endl;
        return APEX_ERROR;
    }

    int hresult = harmony_fetch(tuning_session->hdesc);
    if (hresult < 0) {
        cerr << "Failed to fetch values from server: " << 
                harmony_error_string(tuning_session->hdesc) << endl;
        return APEX_ERROR;
    }
    else if (hresult == 0) {
        /* New values were not available at this time.
         * Bundles remain unchanged by Harmony system.
         */
    }
    else if (hresult > 0) {
        /* The Harmony system modified the variable values.
         * Do any systemic updates to deal with such a change.
         */
    }

    return APEX_NOERROR;
}

#else // APEX_HAVE_ACTIVEHARMONY
int apex_throughput_throttling_ah_policy(apex_context const context) { 
    APEX_UNUSED(context);
    return APEX_NOERROR; 
}
int apex_throughput_tuning_policy(apex_context const context) {
    APEX_UNUSED(context);
    return APEX_NOERROR; 
}
int apex_custom_tuning_policy(apex_context const context) {
    APEX_UNUSED(context);
    return APEX_NOERROR;
}
#endif // APEX_HAVE_ACTIVEHARMONY

/// ----------------------------------------------------------------------------
///
/// Functions to setup and shutdown energy measurements during execution
/// Uses the APEX Policy Engine interface to dynamicly set a variable "apex_throttleOn"
/// that the scheduling code used to determine the best number of active threads
///
/// This probably wants to be ifdef under some flag -- I don't want to figure out
/// how to do this currently AKP 11/01/14
/// ----------------------------------------------------------------------------

inline void __read_common_variables() {
    tuning_session->max_threads = tuning_session->thread_cap = apex::hardware_concurrency();
    tuning_session->min_threads = 1;
    if (apex::apex_options::throttle_concurrency()) {
        apex_checkThrottling = true;
        tuning_session->max_threads = apex::apex_options::throttling_max_threads();
        tuning_session->thread_cap = tuning_session->max_threads;
        tuning_session->min_threads = apex::apex_options::throttling_min_threads();
        apex::apex * instance = apex::apex::instance();
        if (instance != NULL && instance->get_node_id() == 0) {
            cout << "APEX concurrency throttling enabled, min threads: " << tuning_session->min_threads << " max threads: " << tuning_session->max_threads << endl;
        }
    }
}

inline int __setup_power_cap_throttling()
{
    if(tuning_session->apex_energy_init) {
        std::cerr << "power cap throttling already initialized!" << std::endl;
        return APEX_ERROR;
    }
    tuning_session->apex_energy_init = true;
    __read_common_variables();
    // if desired for this execution set up throttling & final print of total energy used 
    if (apex_checkThrottling) {
      tuning_session->max_watts = apex::apex_options::throttling_max_watts();
      tuning_session->min_watts = apex::apex_options::throttling_min_watts();
      if (apex::apex_options::throttle_energy()) {
        apex_energyThrottling = true;
      }
      apex::apex * instance = apex::apex::instance();
      if (instance != NULL && instance->get_node_id() == 0) {
        cout << "APEX periodic throttling for energy savings, min watts: " << tuning_session->min_watts << " max watts: " << tuning_session->max_watts << endl;
      }
      apex::register_periodic_policy(1000000, apex_power_throttling_policy);
      // get an initial power reading
      apex::current_power_high();
#ifdef APEX_HAVE_RCR
      energyDaemonEnter();
#endif
    }
  return APEX_NOERROR;
}

#ifdef APEX_HAVE_ACTIVEHARMONY
inline void __apex_active_harmony_setup(void) {
    static const char* session_name = "APEX Throttling";
    tuning_session->hdesc = harmony_init(NULL, NULL);
    if (tuning_session->hdesc == NULL) {
        cerr << "Failed to initialize Active Harmony" << endl;
        return;
    }
    if (harmony_session_name(tuning_session->hdesc, session_name) != 0) {
        cerr << "Could not set Active Harmony session name" << endl;
        return;
    }
    if (harmony_int(tuning_session->hdesc, "tuning_session->thread_cap", tuning_session->min_threads, tuning_session->max_threads, tuning_session->thread_step) != 0) {
        cerr << "Failed to define Active Harmony tuning session" << endl;
        return;
    }
    if (harmony_strategy(tuning_session->hdesc, "pro.so") != 0) {
        cerr << "Failed to set Active Harmony tuning strategy" << endl;
        return;
    }
    if (harmony_launch(tuning_session->hdesc, NULL, 0) != 0) {
        cerr << "Failed to launch Active Harmony tuning session: " << 
            endl << harmony_error_string(tuning_session->hdesc) << endl;
        return;
    }
        tuning_session->__num_ah_inputs = 1;
        tuning_session->__ah_inputs[0] = &(tuning_session->thread_cap);
    if (harmony_bind_int(tuning_session->hdesc, "tuning_session->thread_cap", &(tuning_session->thread_cap)) != 0) {
        cerr << "Failed to register Active Harmony variable" << endl;
        return;
    }
    if (harmony_join(tuning_session->hdesc, NULL, 0, session_name) != 0) {
        cerr << "Failed to launch Active Harmony tuning session" << endl;
        return;
    }
}

inline void __active_harmony_throughput_setup(int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
    static const char* session_name = "APEX Throttling";
    tuning_session->hdesc = harmony_init(NULL, NULL);
    if (tuning_session->hdesc == NULL) {
        cerr << "Failed to initialize Active Harmony" << endl;
        return;
    }
    if (harmony_session_name(tuning_session->hdesc, session_name) != 0) {
        cerr << "Could not set Active Harmony session name" << endl;
        return;
    }
    if (harmony_strategy(tuning_session->hdesc, "pro.so") != 0) {
        cerr << "Failed to set Active Harmony tuning strategy" << endl;
        return;
    }
    char tmpstr[12] = {0};
        tuning_session->__num_ah_inputs = num_inputs;
    for (int i = 0 ; i < num_inputs ; i++ ) {
        sprintf (tmpstr, "param_%d", i);
        if (harmony_int(tuning_session->hdesc, tmpstr, mins[i], maxs[i], steps[i]) != 0) {
            cerr << "Failed to define Active Harmony tuning session" << endl;
            return;
        }
    }
    if (harmony_launch(tuning_session->hdesc, NULL, 0) != 0) {
        cerr << "Failed to launch Active Harmony tuning session: " << 
            endl << harmony_error_string(tuning_session->hdesc) << endl;
        return;
    }
    for (int i = 0 ; i < num_inputs ; i++ ) {
        sprintf (tmpstr, "param_%d", i);
        if (harmony_bind_int(tuning_session->hdesc, tmpstr, inputs[i]) != 0) {
            cerr << "Failed to register Active Harmony variable" << endl;
            return;
        }
                tuning_session->__ah_inputs[i] = inputs[i];
    }
    if (harmony_join(tuning_session->hdesc, NULL, 0, session_name) != 0) {
        cerr << "Failed to join Active Harmony tuning session" << endl;
        return;
    }
}

inline int __active_harmony_custom_setup(int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
    static const char* session_name = "APEX Custom Tuning";
    tuning_session->hdesc = harmony_init(NULL, NULL);
    if (tuning_session->hdesc == NULL) {
        cerr << "Failed to initialize Active Harmony" << endl;
        return APEX_ERROR;
    }
    if (harmony_session_name(tuning_session->hdesc, session_name) != 0) {
        cerr << "Could not set Active Harmony session name" << endl;
        return APEX_ERROR;
    }

    // TODO: Change strategy to support multi-objective optimization
    // (will need multiple metrics-of-interest)
    if (harmony_strategy(tuning_session->hdesc, "pro.so") != 0) {
        cerr << "Failed to set Active Harmony tuning strategy" << endl;
        return APEX_ERROR;
    }
    char tmpstr[12] = {0};
    for (int i = 0 ; i < num_inputs ; i++ ) {
        sprintf (tmpstr, "param_%d", i);
        if (harmony_int(tuning_session->hdesc, tmpstr, mins[i], maxs[i], steps[i]) != 0) {
            cerr << "Failed to define Active Harmony tuning session" << endl;
            return APEX_ERROR;
        }
    }
    if (harmony_launch(tuning_session->hdesc, NULL, 0) != 0) {
        cerr << "Failed to launch Active Harmony tuning session: " << 
            endl << harmony_error_string(tuning_session->hdesc) << endl;
        return APEX_ERROR;
    }
    for (int i = 0 ; i < num_inputs ; i++ ) {
        sprintf (tmpstr, "param_%d", i);
        tuning_session->tunable_params.push_back(std::make_pair(tmpstr, inputs[i]));
        if (harmony_bind_int(tuning_session->hdesc, tmpstr, inputs[i]) != 0) {
            cerr << "Failed to register Active Harmony variable" << endl;
            return APEX_ERROR;
        }
        tuning_session->__ah_inputs[i] = inputs[i];
    }
    if (harmony_join(tuning_session->hdesc, NULL, 0, session_name) != 0) {
        cerr << "Failed to join Active Harmony tuning session" << endl;
        return APEX_ERROR;
    }

    return APEX_NOERROR;
}

inline void __apex_active_harmony_shutdown(void) {
    /* Leave the session */
    if (harmony_leave(tuning_session->hdesc) != 0) {
        cerr << "Failed to disconnect from harmony session." << endl;;
        return;
    }
    harmony_fini(tuning_session->hdesc);
}

#else
inline void __apex_active_harmony_setup(void) { }
inline void __active_harmony_throughput_setup(int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
  APEX_UNUSED(num_inputs);
  APEX_UNUSED(inputs);
  APEX_UNUSED(mins);
  APEX_UNUSED(maxs);
  APEX_UNUSED(steps);
  std::cerr << "WARNING: Active Harmony setup attempted but APEX was built without Active Harmony support!" << std::endl;
}
inline int __active_harmony_custom_setup(int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
  APEX_UNUSED(num_inputs);
  APEX_UNUSED(inputs);
  APEX_UNUSED(mins);
  APEX_UNUSED(maxs);
  APEX_UNUSED(steps);
  std::cerr << "WARNING: Active Harmony setup attempted but APEX was built without Active Harmony support!" << std::endl;
  return APEX_NOERROR;
}
inline void __apex_active_harmony_shutdown(void) { }
#endif

inline int __common_setup_timer_throttling(apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval)
{
    __read_common_variables();
    if (apex::apex_options::throttle_concurrency()) {
        tuning_session->function_history.calls = 0.0;
        tuning_session->function_history.accumulated = 0.0;
        tuning_session->function_baseline.calls = 0.0;
        tuning_session->function_baseline.accumulated = 0.0;
        tuning_session->throttling_criteria = criteria;
        tuning_session->evaluations = (double*)(calloc(tuning_session->max_threads+1, sizeof(double)));
        tuning_session->observations = (int*)(calloc(tuning_session->max_threads+1, sizeof(int)));
        apex::apex * instance = apex::apex::instance();
        if (instance != NULL && instance->get_node_id() == 0) {
            tuning_session->cap_data.open("tuning_session->cap_data.dat");
            tuning_session->cap_data_open = true;
        }
        if (method == APEX_SIMPLE_HYSTERESIS) {
            apex::register_periodic_policy(update_interval, apex_throughput_throttling_policy);
        } else if (method == APEX_DISCRETE_HILL_CLIMBING) {
            apex::register_periodic_policy(update_interval, apex_throughput_throttling_dhc_policy);
        } else if (method == APEX_ACTIVE_HARMONY) {
            __apex_active_harmony_setup();
            apex::register_periodic_policy(update_interval, apex_throughput_throttling_ah_policy);
        }
    }
    return APEX_NOERROR;
}

inline int __common_setup_throughput_tuning(apex_optimization_criteria_t criteria,
        apex_event_type event_type, int num_inputs, long ** inputs, long * mins,
        long * maxs, long * steps)
{
    __read_common_variables();
    if (apex::apex_options::throttle_concurrency()) {
        tuning_session->function_history.calls = 0.0;
        tuning_session->function_history.accumulated = 0.0;
        tuning_session->function_baseline.calls = 0.0;
        tuning_session->function_baseline.accumulated = 0.0;
        tuning_session->throttling_criteria = criteria;
        tuning_session->evaluations = (double*)(calloc(tuning_session->max_threads+1, sizeof(double)));
        tuning_session->observations = (int*)(calloc(tuning_session->max_threads+1, sizeof(int)));
        __active_harmony_throughput_setup(num_inputs, inputs, mins, maxs, steps);
        apex::register_policy(event_type, apex_throughput_tuning_policy);
    }
    return APEX_NOERROR;
}

inline int __common_setup_custom_tuning( apex_event_type event_type, int num_inputs,
        long ** inputs, long * mins, long * maxs, long * steps)
{
    __read_common_variables();
    int status = __active_harmony_custom_setup(num_inputs, inputs, mins, maxs, steps);
    if(status == APEX_NOERROR) {
        apex::register_policy(event_type, apex_custom_tuning_policy);
    }
    return status;
}

inline int __setup_throughput_tuning(apex_function_address the_address,
        apex_optimization_criteria_t criteria, apex_event_type event_type, 
        int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
    tuning_session->function_of_interest = the_address;
    return __common_setup_throughput_tuning(criteria, event_type, num_inputs, inputs, mins, maxs, steps);
}

inline int __setup_throughput_tuning(std::string &the_name,
        apex_optimization_criteria_t criteria, apex_event_type event_type, 
        int num_inputs, long ** inputs, long * mins, long * maxs, long * steps) {
    tuning_session->function_name_of_interest = string(the_name);
    return __common_setup_throughput_tuning(criteria, event_type, num_inputs, inputs, mins, maxs, steps);
}

inline int __setup_custom_tuning(std::function<double()> metric,
        apex_event_type event_type, int num_inputs, long ** inputs,
        long * mins, long * maxs, long * steps) {
    tuning_session->metric_of_interest = metric;
    return __common_setup_custom_tuning(event_type, num_inputs, inputs, mins, maxs, steps);
}

inline int __setup_timer_throttling(apex_function_address the_address, apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval)
{
    tuning_session->function_of_interest = the_address;
    return __common_setup_timer_throttling(criteria, method, update_interval);
}

inline int __setup_timer_throttling(const string& the_name, apex_optimization_criteria_t criteria,
        apex_optimization_method_t method, unsigned long update_interval)
{
    if(tuning_session->apex_timer_init) {
        std::cerr << "timer throttling already initialized!" << std::endl;
        return APEX_ERROR;
    }
    tuning_session->apex_timer_init = true;
    if (the_name == "") {
        fprintf(stderr, "Timer/counter name for throttling is undefined. Please specify a name.\n");
        abort();
    }
    tuning_session->function_name_of_interest = the_name;
#ifdef APEX_DEBUG_THROTTLE
    std::cerr << "Setting up timer throttling for " << the_name << std::endl;
#endif
    return __common_setup_timer_throttling(criteria, method, update_interval);
}

inline int __shutdown_throttling(void)
{
    apex_final = true;
  //printf("periodic_policy called %d times\n", tuning_session->test_pp);
    if (tuning_session->cap_data_open) {
        tuning_session->cap_data_open = false;
        tuning_session->cap_data.close();
    }
  return APEX_NOERROR;
}

/* These are the external API versions of the above functions. */

namespace apex {

APEX_EXPORT int setup_power_cap_throttling(void) {
    return __setup_power_cap_throttling();
}

APEX_EXPORT int setup_timer_throttling(apex_function_address the_address,
        apex_optimization_criteria_t criteria, apex_optimization_method_t method,
        unsigned long update_interval) {
    return __setup_timer_throttling(the_address, criteria, method, update_interval);
}

APEX_EXPORT int setup_timer_throttling(const std::string &the_name,
        apex_optimization_criteria_t criteria, apex_optimization_method_t method,
        unsigned long update_interval) {
    return __setup_timer_throttling(the_name, criteria, method, update_interval);
}

APEX_EXPORT int setup_throughput_tuning(apex_function_address the_address,
        apex_optimization_criteria_t criteria, apex_event_type event_type, int num_inputs,
        long ** inputs, long * mins, long * maxs, long * steps) {
    return __setup_throughput_tuning(the_address, criteria, event_type, num_inputs, inputs, mins, maxs, steps);
}

APEX_EXPORT int setup_throughput_tuning(std::string &the_name,
        apex_optimization_criteria_t criteria, apex_event_type event_type, int num_inputs,
        long ** inputs, long * mins, long * maxs, long * steps) {
    return __setup_throughput_tuning(the_name, criteria, event_type, num_inputs, inputs, mins, maxs, steps);
}

APEX_EXPORT int setup_custom_tuning(std::function<double()> metric,
        apex_event_type event_type, int num_inputs, long ** inputs,
        long * mins, long * maxs, long * steps) {
    return __setup_custom_tuning(metric, event_type, num_inputs, inputs, mins, maxs, steps);
}


APEX_EXPORT int shutdown_throttling(void) {
    return __shutdown_throttling();
}

APEX_EXPORT int get_thread_cap(void) {
    return __get_thread_cap();
}

APEX_EXPORT int get_input2(void) {
    return (int)*(tuning_session->__ah_inputs[1]);
}


APEX_EXPORT std::vector<std::pair<std::string,long*>> & get_tunable_params() {
    return tuning_session->tunable_params;
}

}

extern "C" {

APEX_EXPORT int apex_setup_power_cap_throttling(void) {
    return __setup_power_cap_throttling();
}

APEX_EXPORT int apex_setup_timer_throttling(apex_profiler_type type, void * identifier,
        apex_optimization_criteria_t criteria, 
        apex_optimization_method_t method, unsigned long update_interval) {
    assert(identifier);
    if (type == APEX_FUNCTION_ADDRESS) {
        apex_function_address the_address = (apex_function_address)identifier;
        return __setup_timer_throttling(the_address, criteria, method, update_interval);
    }
    if (type == APEX_NAME_STRING) {
        string tmp((const char *)identifier);
        return __setup_timer_throttling(tmp, criteria, method, update_interval);
    }
    return APEX_ERROR;
}

APEX_EXPORT int apex_setup_throughput_tuning( apex_profiler_type type, void * identifier,
        apex_optimization_criteria_t criteria, apex_event_type event_type, int num_inputs,
        long ** inputs, long * mins, long * maxs, long * steps) {
    assert(identifier);
    if (type == APEX_FUNCTION_ADDRESS) {
        return __setup_throughput_tuning((apex_function_address)identifier, criteria, event_type, num_inputs, inputs, mins, maxs, steps);
    } else if (type == APEX_NAME_STRING) {
        string tmp((const char *)identifier);
        return __setup_throughput_tuning(tmp, criteria, event_type, num_inputs, inputs, mins, maxs, steps);
    }
    return APEX_ERROR;
}

APEX_EXPORT int apex_shutdown_throttling(void) {
    return __shutdown_throttling();
}

APEX_EXPORT int apex_get_thread_cap(void) {
    return __get_thread_cap();
}

APEX_EXPORT void apex_set_thread_cap(int new_cap) {
    return __set_thread_cap(new_cap);
}

} // extern "C"

