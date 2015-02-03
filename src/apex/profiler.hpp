#ifndef PROFILER_HPP
#define PROFILER_HPP

//#include <boost/timer/timer.hpp>
#include <chrono>
#include <iostream>
#include <math.h>
#include "apex_types.h"

using namespace std;
using namespace std::chrono;

namespace apex {

class profiler {
public:
        //boost::timer::cpu_timer t; // starts the timer when profiler is constructed!
	high_resolution_clock::time_point start;
	high_resolution_clock::time_point end;
#if APEX_HAVE_PAPI
	long long papi_start_values[8];
	long long papi_stop_values[8];
#endif
    double value;
    void * action_address;
    string * timer_name;
    bool have_name;
    bool is_counter;
    bool is_resume;
    bool is_reset;
    profiler(void * address, bool resume = false, bool reset = false) : 
	    start(high_resolution_clock::now()), 
	    action_address(address), 
	    have_name(false), 
	    is_counter(false),
        is_resume(resume),
        is_reset(reset) {};
    profiler(string * name, bool resume = false, bool reset = false) : 
	    start(high_resolution_clock::now()), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(false),
        is_resume(resume),
        is_reset(reset) {};
    profiler(string * name, double value_) : 
	    value(value_), 
	    timer_name(name), 
	    have_name(true), 
	    is_counter(true),
        is_resume(false),
        is_reset(false) { }; 
    ~profiler(void) { if (have_name) delete timer_name; };
    void stop(void) {
        end = high_resolution_clock::now();
	};
	double elapsed(void) {
        if(is_counter) {
            return value;
        } else {
            duration<double> time_span = duration_cast<duration<double>>(end - start);
            return time_span.count();
        }
	}
};

class profile {
private:
	apex_profile _profile;
public:
	profile(double initial, apex_profile_type type = TIMER) {
        _profile.type = type;
		_profile.calls = 1.0;
		_profile.accumulated = initial;
		_profile.sum_squares = initial*initial;
		_profile.minimum = initial;
		_profile.maximum = initial;
	};
	void increment(double increase) {
		_profile.accumulated += increase;
		_profile.sum_squares += (increase * increase);
		_profile.minimum = _profile.minimum > increase ? increase : _profile.minimum;
		_profile.maximum = _profile.maximum < increase ? increase : _profile.maximum;
		_profile.calls = _profile.calls + 1.0;
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
