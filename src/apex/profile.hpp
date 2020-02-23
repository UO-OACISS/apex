//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"

// Use this if you want the min, max and stddev.
#define FULL_STATISTICS

namespace apex {

class profile {
private:
    apex_profile _profile;
public:
    profile(double initial, int num_metrics, double * papi_metrics, bool
        yielded = false, apex_profile_type type = APEX_TIMER) {
        _profile.type = type;
        if (!yielded) {
            _profile.calls = 1.0;
        } else {
            _profile.calls = 0.0;
        }
        _profile.accumulated = initial;
        for (int i = 0 ; i < num_metrics ; i++) {
            _profile.papi_metrics[i] = papi_metrics[i];
        }
#ifdef FULL_STATISTICS
        _profile.sum_squares = initial*initial;
        _profile.minimum = initial;
        _profile.maximum = initial;
#endif
    };
    void increment(double increase, int num_metrics, double * papi_metrics,
        bool yielded) {
        _profile.accumulated += increase;
        for (int i = 0 ; i < num_metrics ; i++) {
            _profile.papi_metrics[i] += papi_metrics[i];
        }
#ifdef FULL_STATISTICS
        _profile.sum_squares += (increase * increase);
        // if not a fully completed task, don't modify these until it is done
        _profile.minimum = _profile.minimum > increase ? increase : _profile.minimum;
        _profile.maximum = _profile.maximum < increase ? increase : _profile.maximum;
#endif
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
        _profile.times_reset++;
    };
    double get_calls() { return _profile.calls; }
    double get_mean() { return (_profile.accumulated / _profile.calls); }
    double get_accumulated() { return (_profile.accumulated); }
    double * get_papi_metrics() { return (_profile.papi_metrics); }
    double get_minimum() { return (_profile.minimum); }
    double get_maximum() { return (_profile.maximum); }
    int get_times_reset() { return (_profile.times_reset); }
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

