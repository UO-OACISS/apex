#ifndef PROFILE_HPP
#define PROFILE_HPP

//#include <boost/timer/timer.hpp>
#include <chrono>
#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"

namespace apex {

class profile {
private:
	apex_profile _profile;
public:
	profile(double initial, bool yielded = false, apex_profile_type type = APEX_TIMER) {
        _profile.type = type;
        if (!yielded) {
		    _profile.calls = 1.0;
        } else {
            _profile.calls = 0.0;
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
		double variance = ((_profile.sum_squares / _profile.calls) - (mean * mean));
        return variance >= 0.0 ? variance : 0.0;
	}
    double get_sum_squares() { return _profile.sum_squares; }
	double get_stddev() { return sqrt(get_variance()); }
    apex_profile_type get_type() { return _profile.type; }
	apex_profile * get_profile() { return &_profile; };

};


}

#endif
