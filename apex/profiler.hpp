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
	//boost::timer::cpu_times elapsed_time;
        void * action_address;
        string * timer_name;
	bool have_name;
        profiler(void * address) : start(high_resolution_clock::now()), action_address(address), have_name(false) {};
        profiler(string * name) : start(high_resolution_clock::now()), timer_name(name), have_name(true) {};
        ~profiler(void) {};
        void stop(void) {
		//t.stop();
		end = high_resolution_clock::now();
		//elapsed_time = t.elapsed();
		//cout << t.format(boost::timer::default_places);
	};
	double elapsed(void) {
		duration<double> time_span = duration_cast<duration<double>>(end - start);
		return time_span.count();
	}
};

class profile {
private:
	apex_profile _profile;
public:
	profile(double elapsed) { 
		_profile.calls = 1.0;
		_profile.accumulated_time = elapsed;
		_profile.sum_squares = elapsed*elapsed;
		_profile.minimum = elapsed;
		_profile.maximum = elapsed;
	};
	void increment(double elapsed) {
		_profile.accumulated_time += elapsed;
		_profile.sum_squares += (elapsed * elapsed);
		_profile.minimum = _profile.minimum > elapsed ? elapsed : _profile.minimum;
		_profile.maximum = _profile.maximum < elapsed ? elapsed : _profile.maximum;
		_profile.calls = _profile.calls + 1.0;
	}
	void increment_resume(double elapsed) {
		_profile.accumulated_time += elapsed;
		// how to handle this?
		/*
		sum_squares += (elapsed * elapsed);
		minimum = minimum > elapsed ? elapsed : minimum;
		maximum = maximum < elapsed ? elapsed : maximum;
		*/
	}
	double get_calls() { return _profile.calls; }
	double get_mean() { return (_profile.accumulated_time / _profile.calls); }
	double get_accumulated() { return (_profile.accumulated_time); }
	double get_minimum() { return (_profile.minimum); }
	double get_maximum() { return (_profile.maximum); }
	double get_variance() {
		double mean = get_mean();
		return ((_profile.sum_squares / _profile.calls) - (mean * mean));
	}
	double get_stddev() { return sqrt(get_variance()); }
	apex_profile * get_profile() { return &_profile; };
};

}

#endif
