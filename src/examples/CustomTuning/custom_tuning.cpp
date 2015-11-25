#include <iostream>
#include <array>
#include <algorithm>
#include <functional>
#include "apex_api.hpp"

apex_event_type my_custom_event = APEX_CUSTOM_EVENT_1;

long param_1 = 5;
long param_2 = 5;
double value = 0.0;
int num_iterations = 100;

/**
 * The Main function
 */
int main (int argc, char ** argv) {
    apex::init(argc, argv, "Custom Tuning Test");
    apex::set_node_id(0);

#ifdef APEX_HAVE_ACTIVEHARMONY
    int num_inputs = 2; 
    long * inputs[2] = {0L,0L};
    long mins[2] = {0,0};    // all minimums are 1
    long maxs[2] = {10,10};    // we'll set these later
    long steps[2] = {1,1};   // all step sizes are 1
    inputs[0] = &param_1;
    inputs[1] = &param_2;
    my_custom_event = apex::register_custom_event("Adjust Params");
    std::function<double(void)> func = []()->double{ return value; }; 
    apex_tuning_session_handle session =
        apex::setup_custom_tuning(func, my_custom_event, num_inputs, inputs, mins, maxs, steps);
#else
    std::cerr << "Active Harmony not enabled" << std::endl;
#endif
    std::cout << "Running custom tuning test" << std::endl;

    std::cerr << "Tuning session handle: " << session << std::endl;


    for (int i = 0 ; i < num_iterations ; i++) {
        value = (10 * param_1) - (2 * param_2);
        apex::custom_event(my_custom_event, NULL);
        std::cout << "p1 = " << param_1 << " p2 = " << param_2 << " value = " << value << std::endl;
    }
    std::cout << "done." << std::endl;
#ifdef APEX_HAVE_ACTIVEHARMONY
    if(param_1 != 5 || param_2 != 5) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }
#else
    std::cout << "Test passed (but APEX was built without Active Harmony.)." << std::endl;
#endif
    apex::finalize();
}
