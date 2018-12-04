#include <iostream>
#include <array>
#include <algorithm>
#include <functional>
#include <memory>
#include <list>
#include "apex_api.hpp"
#include "apex_policies.hpp"

apex_event_type my_custom_event = APEX_CUSTOM_EVENT_1;


/**
 * The Main function
 */
int main (int argc, char ** argv) {
    apex::init("Custom Tuning Test", 0, 1);

#ifdef APEX_HAVE_ACTIVEHARMONY

    apex_tuning_request request("tuning_request_example");
    
    // Trigger
    my_custom_event = apex::register_custom_event("Get New Params");
    request.set_trigger(my_custom_event);

    std::shared_ptr<apex_param_long> param_long =
        request.add_param_long("long", 5, 0, 20, 1);
    
    std::shared_ptr<apex_param_double> param_double =
        request.add_param_double("double", 5.0, 0.0, 20.0, 1.0);
    
    std::list<std::string> enum_vals{"a", "b", "c", "d", "e"};
    std::shared_ptr<apex_param_enum> param_enum =
        request.add_param_enum("enum", "c", enum_vals);

    double value = 0.0;
    
    std::function<double(void)> func = [&]()->double{
        return value;
    };
    request.set_metric(func);

    apex_tuning_session_handle session = apex::setup_custom_tuning(request);
    (void)session; // ignore unused warning
    bool exhaustive = false;

    for(int i = 0; i < 150; ++i) {
        apex::profiler * p = apex::start("Iteration");
        std::string s = param_enum->get_value();
        double x = 0.0;
        long y = param_long->get_value();
        double z = param_double->get_value();
        if(s == "a") {
            x = 1.0;
        } else if(s == "b") {
            x = 2.0;
        } else if(s == "c") {
            x = 3.0;
        } else if(s == "d") {
            x = -1.0;
            exhaustive = true;
        } else if(s == "e") {
            x = -1.0;
            exhaustive = true;
        }

        value = x*(y+z);
        std::cerr << "long = " << y << ", double = " << z << ", enum = " << s << ". value = " << value << std::endl;
        apex::stop(p);
        apex::custom_event(my_custom_event, NULL);
    }


#endif

#ifdef APEX_HAVE_ACTIVEHARMONY
    if(value < 0 || !exhaustive) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }
#else
    std::cout << "Test passed (but APEX was built without Active Harmony.)." << std::endl;
#endif
    apex::finalize();
}
