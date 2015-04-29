#ifndef PROFILER_HPP
#define PROFILER_HPP

//#include <boost/timer/timer.hpp>
#include <chrono>
#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"

#define CLOCK_TYPE steady_clock
//#define CLOCK_TYPE high_resolution_clock

namespace apex {

enum struct reset_type {
    NONE, CURRENT, ALL    
};

class profiler {
public:
        //boost::timer::cpu_timer t; // starts the timer when profiler is constructed!
    std::chrono::CLOCK_TYPE::time_point start;
    std::chrono::CLOCK_TYPE::time_point end;
#if APEX_HAVE_PAPI
	long long papi_start_values[8];
	long long papi_stop_values[8];
#endif
    double value;
    apex_function_address action_address;
    std::string * timer_name;
    bool have_name;
    bool is_counter;
    bool is_resume; // for yield or resume
    bool is_elapsed;
    bool safe_to_delete;
    reset_type is_reset;
    profiler(apex_function_address address, 
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(std::chrono::CLOCK_TYPE::now()), 
      value(0.0),
	    action_address(address), 
	    timer_name(NULL), 
	    have_name(false), 
	    is_counter(false),
        is_resume(resume),
        is_elapsed(false),
//#ifdef APEX_HAVE_TAU
        safe_to_delete(false),
//#else
        //safe_to_delete(true),
//#endif
        is_reset(reset) {};
    profiler(std::string * name, 
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(std::chrono::CLOCK_TYPE::now()), 
	    value(0.0), 
	    action_address(0L), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(false),
        is_resume(resume),
        is_elapsed(false),
//#ifdef APEX_HAVE_TAU
        safe_to_delete(false),
//#else
        //safe_to_delete(true),
//#endif
        is_reset(reset) {};
    profiler(std::string * name, double value_) : 
	    value(value_), 
	    action_address(0L), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(true),
        is_resume(false),
        is_elapsed(true),
//#ifdef APEX_HAVE_TAU
        safe_to_delete(false),
//#else
        //safe_to_delete(true),
//#endif
        is_reset(reset_type::NONE) { }; 
    ~profiler(void) { if (have_name) delete timer_name; };
    void stop(bool is_resume = false) {
        this->is_resume = is_resume;
        end = std::chrono::CLOCK_TYPE::now();
	};
	double elapsed(void) {
        if(is_counter || is_elapsed) {
            return value;
        } else {
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            return time_span.count();
        }
	}
};

}

#endif
