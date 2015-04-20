#ifndef PROFILER_HPP
#define PROFILER_HPP

//#include <boost/timer/timer.hpp>
#include <chrono>
#include <iostream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"

namespace apex {

enum struct reset_type {
    NONE, CURRENT, ALL    
};

class profiler {
public:
        //boost::timer::cpu_timer t; // starts the timer when profiler is constructed!
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
#if APEX_HAVE_PAPI
	long long papi_start_values[8];
	long long papi_stop_values[8];
#endif
    double value;
    apex_function_address action_address;
    std::string * timer_name;
    bool have_name;
    bool is_counter;
    bool is_resume;
    bool safe_to_delete;
    reset_type is_reset;
    profiler(apex_function_address address, 
             const std::chrono::high_resolution_clock::time_point &timestamp, 
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(timestamp), 
      value(0.0),
	    action_address(address), 
	    timer_name(NULL), 
	    have_name(false), 
	    is_counter(false),
        is_resume(resume),
//#ifdef APEX_HAVE_TAU
        safe_to_delete(false),
//#else
        //safe_to_delete(true),
//#endif
        is_reset(reset) {};
    profiler(std::string * name, 
             const std::chrono::high_resolution_clock::time_point &timestamp, 
             bool resume = false, 
             reset_type reset = reset_type::NONE) : 
	    start(timestamp), 
	    value(0.0), 
	    action_address(0L), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(false),
        is_resume(resume),
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
//#ifdef APEX_HAVE_TAU
        safe_to_delete(false),
//#else
        //safe_to_delete(true),
//#endif
        is_reset(reset_type::NONE) { }; 
    ~profiler(void) { if (have_name) delete timer_name; };
    void stop(const std::chrono::high_resolution_clock::time_point &timestamp) {
        end = timestamp;
	};
	double elapsed(void) {
        if(is_counter) {
            return value;
        } else {
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            return time_span.count();
        }
	}
};

class profile {
private:
	apex_profile _profile;
public:
	profile(double initial, bool yielded = false, apex_profile_type type = APEX_TIMER) {
        _profile.type = type;
        if (!yielded) {
		    _profile.calls = 1.0;
        }
		_profile.accumulated = initial;
		_profile.sum_squares = initial*initial;
		_profile.minimum = initial;
		_profile.maximum = initial;
	};
	void increment(double increase, bool yielded) {
		_profile.accumulated += increase;
		_profile.sum_squares += (increase * increase);
        // if not a fully completed task, don't modify these until it is done
		_profile.minimum = _profile.minimum > increase ? increase : _profile.minimum;
		_profile.maximum = _profile.maximum < increase ? increase : _profile.maximum;
        if (!yielded) {
		  _profile.calls = _profile.calls + 1.0;
        } 
	}
	void increment_resume(double increase) {
		_profile.accumulated += increase;
		// how to handle this?
		/*
		sum_squares += (elapsed * elapsed);
		minimum = minimum > elapsed ? elapsed : minimum;
		maximum = maximum < elapsed ? elapsed : maximum;
		*/
	}
    void reset() {
        _profile.calls = 0.0;
        _profile.accumulated = 0.0;
        _profile.sum_squares = 0.0;
        _profile.minimum = 0.0;
        _profile.maximum = 0.0;
    };
	double get_calls() { return _profile.calls; }
	double get_mean() { return (_profile.accumulated / _profile.calls); }
	double get_accumulated() { return (_profile.accumulated); }
	double get_minimum() { return (_profile.minimum); }
	double get_maximum() { return (_profile.maximum); }
	double get_variance() {
		double mean = get_mean();
		return ((_profile.sum_squares / _profile.calls) - (mean * mean));
	}
    double get_sum_squares() { return _profile.sum_squares; }
	double get_stddev() { return sqrt(get_variance()); }
    apex_profile_type get_type() { return _profile.type; }
	apex_profile * get_profile() { return &_profile; };
};

}

#endif
