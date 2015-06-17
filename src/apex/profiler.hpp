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

class disabled_profiler_exception : public std::exception {
    virtual const char* what() const throw() {
      return "Disabled profiler.";
    }
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
    double children_value;
    apex_function_address action_address;
    std::string * timer_name;
    bool have_name;
    bool is_counter;
    bool is_resume; // for yield or resume
    reset_type is_reset;
    bool stopped;
    profiler(apex_function_address address,
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(std::chrono::CLOCK_TYPE::now()), 
        value(0.0),
        children_value(0.0),
	    action_address(address), 
	    timer_name(nullptr), 
	    have_name(false), 
	    is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false) {};
    profiler(std::string * name, 
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(std::chrono::CLOCK_TYPE::now()), 
	    value(0.0), 
        children_value(0.0),
	    action_address(0L), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false) {};
    profiler(std::string * name, double value_) : 
	    value(value_), 
        children_value(0.0),
	    action_address(0L), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(true),
        is_resume(false),
        is_reset(reset_type::NONE), stopped(true) { }; 
    //copy constructor
    profiler(profiler* in) : start(in->start), end(in->end) {
    //start = in->start;
    //end = in->start;
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < 8 ; i++) {
	        papi_start_values[i] = in->papi_start_values[i];
	        papi_stop_values[i] = in->papi_stop_values[i];
        }
#endif
    value = in->elapsed();
    children_value = in->children_value;
    action_address = in->action_address;
    have_name = in->have_name;
    if (have_name && in->timer_name != nullptr) {
      timer_name = new std::string(in->timer_name->c_str());
    }
    is_counter = in->is_counter;
    is_resume = in->is_resume; // for yield or resume
    is_reset = in->is_reset;
    stopped = in->stopped;
    }
    ~profiler(void) { if (have_name && timer_name != nullptr) delete timer_name; };
    // for "yield" support
    void stop(bool is_resume) {
        this->is_resume = is_resume;
        end = std::chrono::CLOCK_TYPE::now();
        stopped = true;
	};
    void stop() {
      end = std::chrono::CLOCK_TYPE::now();
      stopped = true;
	};
	double elapsed(void) {
        if(is_counter) {
            return value;
        } else {
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            return time_span.count();
        }
	}
    double exclusive_elapsed(void) {
        return elapsed() - children_value;
    }

    static inline profiler* get_disabled_profiler(void) { 
        return disabled_profiler;
    }
    // default constructor for the dummy profiler
    profiler(void) {};
    // dummy profiler to indicate that stop/yield should resume immediately
    static profiler* disabled_profiler; // initialized in profiler_listener.cpp
};

}

#endif
