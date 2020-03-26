#include <iostream>
#include <array>
#include <algorithm>
#include <functional>
#include "apex_api.hpp"

apex_event_type my_custom_event = APEX_CUSTOM_EVENT_1;
apex_event_type my_custom_event_2 = APEX_CUSTOM_EVENT_2;

long param_1 = 5;
long param_2 = 5;
double value = 0.0;
int num_iterations = 100;

long x = 1;
double sv = 0.0;

/**
 * The Main function
 */
int main (int argc, char ** argv) {
    APEX_UNUSED(argc);
    APEX_UNUSED(argv);
    apex::init("Custom Tuning Test", 0, 1);

#ifdef APEX_HAVE_ACTIVEHARMONY
    int num_inputs = 2;
    long * inputs[2] = {0L,0L};
    long mins[2] = {0,0};    // all minimums are 1
    long maxs[2] = {10,10};    // we'll set these later
    long steps[2] = {1,1};   // all step sizes are 1
    inputs[0] = &param_1;
    inputs[1] = &param_2;
    my_custom_event = apex::register_custom_event("Adjust Eqn1 Params");
    std::function<double(void)> func = []()->double{ return value; };
    apex_tuning_session_handle session =
        apex::setup_custom_tuning(func, my_custom_event, num_inputs, inputs, mins, maxs, steps);
    std::cout << "Running custom tuning test" << std::endl;

    int num_inputs_2 = 1;
    long * inputs_2[1] = {0L};
    long mins_2[1] = {1};
    long maxs_2[1] = {100};
    long steps_2[1] = {1};
    inputs_2[0] = &x;
    my_custom_event_2 = apex::register_custom_event("Adjust Eqn2 Params");
    std::function<double(void)> func_2 = []()->double{ return sv; };
    apex_tuning_session_handle session_2 =
        apex::setup_custom_tuning(func_2, my_custom_event_2, num_inputs_2, inputs_2, mins_2, maxs_2, steps_2);

    std::cerr << "Tuning session 1 handle: " << session << std::endl;
    std::cerr << "Tuning session 2 handle: " << session_2 << std::endl;

#else
    std::cerr << "Active Harmony not enabled" << std::endl;
#endif

    for (int i = 0 ; i < num_iterations ; i++) {
        value = (10 * param_1) - (2 * param_2);
        apex::custom_event(my_custom_event, NULL);
        std::cout << "p1 = " << param_1 << " p2 = " << param_2 << " value = " << value << std::endl;
        sv = sin(sqrt(x));
        apex::custom_event(my_custom_event_2, NULL);
        std::cout << "x = " << x << " sv = " << sv << std::endl;
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
