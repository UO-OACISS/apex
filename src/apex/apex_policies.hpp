#ifndef APEX_POLICIES_HPP
#define APEX_POLICIES_HPP

#include "apex_api.hpp"
#include "apex.hpp"
#include "apex_export.h"
#include "utils.hpp"
#include <stdint.h>
#include <fstream>
#include <boost/atomic.hpp>

#ifdef APEX_HAVE_ACTIVEHARMONY
#include "hclient.h"
#endif

// include the C API
#include "apex_policies.h"

struct apex_tuning_session {
    apex_tuning_session_handle id;

#ifdef APEX_HAVE_ACTIVEHARMONY
    hdesc_t * hdesc;
#else
    void * hdesc;
#endif

    int test_pp = 0;
    boost::atomic<bool> apex_energy_init{false};
    boost::atomic<bool> apex_timer_init{false};

    bool converged_message = false;

    // variables related to power throttling
    double max_watts = APEX_HIGH_POWER_LIMIT;
    double min_watts = APEX_LOW_POWER_LIMIT;
    int max_threads = APEX_MAX_THREADS;
    int min_threads = APEX_MIN_THREADS;
    int thread_step = 1;
    long int thread_cap = apex::hardware_concurrency();
    double moving_average = 0.0;
    int window_size = MAX_WINDOW_SIZE;
    int delay = 0;
    int throughput_delay = MAX_WINDOW_SIZE;

    // variables related to throughput or custom throttling
    apex_function_address function_of_interest = APEX_NULL_FUNCTION_ADDRESS;
    std::string function_name_of_interest = "";
    std::function<double()> metric_of_interest;
    apex_profile function_baseline;
    apex_profile function_history;
    last_action_t last_action = INITIAL_STATE;
    apex_optimization_criteria_t throttling_criteria = APEX_MAXIMIZE_THROUGHPUT;
    std::vector<std::pair<std::string,long*>> tunable_params;

    // variables for hill climbing
    double * evaluations = NULL;
    int * observations = NULL;
    std::ofstream cap_data;
    bool cap_data_open = false;

    // variables for active harmony general tuning
    long int *__ah_inputs[10]; // more than 10 would be pointless
    int __num_ah_inputs;

    apex_tuning_session(apex_tuning_session_handle h) : id{h} {};
};


#endif // APEX_POLICIES_HPP
