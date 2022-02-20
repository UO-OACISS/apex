/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"
#include "string.h"

// Use this if you want the min, max and stddev.
#define FULL_STATISTICS

namespace apex {

class profile {
private:
    apex_profile _profile;
    /* Ideally, each value in _profile would be an atomic.  however,
     * we have to return the apex_profile type to C through the API,
     * so it has to be a C type.  Instead, each instance of the profile
     * class will have its own mutex, controlling access to the data in
     * _profile.  Only needed when updating the values. */
    std::mutex _mtx;
public:
    profile(double initial, int num_metrics, double * papi_metrics, bool
        yielded = false, apex_profile_type type = APEX_TIMER) {
        memset(&(this->_profile), 0, sizeof(apex_profile));
        _profile.type = type;
        if (!yielded) {
            _profile.calls = 1.0;
        } else {
            _profile.calls = 0.0;
        }
        _profile.stops = 1.0;
        _profile.accumulated = initial;
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < num_metrics ; i++) {
            _profile.papi_metrics[i] = papi_metrics[i];
        }
#else
        APEX_UNUSED(num_metrics);
        APEX_UNUSED(papi_metrics);
#endif
        _profile.times_reset = 0;
#ifdef FULL_STATISTICS
        _profile.sum_squares = initial*initial;
        _profile.minimum = initial;
        _profile.maximum = initial;
#endif
        _profile.allocations = 0;
        _profile.frees = 0;
        _profile.bytes_allocated = 0;
        _profile.bytes_freed = 0;
    };
    profile(double initial, int num_metrics, double * papi_metrics, bool
        yielded, double allocations, double frees, double bytes_allocated,
        double bytes_freed) {
        _profile.type = APEX_TIMER;
        if (!yielded) {
            _profile.calls = 1.0;
        } else {
            _profile.calls = 0.0;
        }
        _profile.stops = 1.0;
        _profile.accumulated = initial;
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < num_metrics ; i++) {
            _profile.papi_metrics[i] = papi_metrics[i];
        }
#else
        APEX_UNUSED(num_metrics);
        APEX_UNUSED(papi_metrics);
#endif
        _profile.times_reset = 0;
#ifdef FULL_STATISTICS
        _profile.sum_squares = initial*initial;
        _profile.minimum = initial;
        _profile.maximum = initial;
#endif
        _profile.allocations = allocations;
        _profile.frees = frees;
        _profile.bytes_allocated = bytes_allocated;
        _profile.bytes_freed = bytes_freed;
    };
    /* This constructor is so that we can create a dummy wrapper around profile
     * data after we've done a reduction across ranks. */
    profile(apex_profile * values) {
        memcpy(&_profile, values, sizeof(apex_profile));
    }
    void increment(double increase, int num_metrics, double * papi_metrics,
        bool yielded) {
        _mtx.lock();
        _profile.accumulated += increase;
        _profile.stops = _profile.stops + 1.0;
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < num_metrics ; i++) {
            _profile.papi_metrics[i] += papi_metrics[i];
        }
#else
        APEX_UNUSED(num_metrics);
        APEX_UNUSED(papi_metrics);
#endif
#ifdef FULL_STATISTICS
        _profile.sum_squares += (increase * increase);
        // if not a fully completed task, don't modify these until it is done
        _profile.minimum = _profile.minimum > increase ? increase : _profile.minimum;
        _profile.maximum = _profile.maximum < increase ? increase : _profile.maximum;
#endif
        if (!yielded) {
          _profile.calls = _profile.calls + 1.0;
        }
        _mtx.unlock();
    }
    void increment(double increase, int num_metrics, double * papi_metrics,
        double allocations, double frees, double bytes_allocated, double bytes_freed,
        bool yielded) {
        increment(increase, num_metrics, papi_metrics, yielded);
        _mtx.lock();
        _profile.allocations += allocations;
        _profile.frees += frees;
        _profile.bytes_allocated += bytes_allocated;
        _profile.bytes_freed += bytes_freed;
        _mtx.unlock();
    }
    void reset() {
        _mtx.lock();
        _profile.calls = 0.0;
        _profile.stops = 0.0;
        _profile.accumulated = 0.0;
        _profile.sum_squares = 0.0;
        _profile.minimum = 0.0;
        _profile.maximum = 0.0;
        _profile.times_reset++;
        _mtx.unlock();
    };
    double get_calls() { return _profile.calls; }
    double get_stops() { return _profile.stops; }
    double get_mean() {
        return (get_accumulated() / _profile.calls);
    }
    double get_mean_useconds() {
        return (get_accumulated_useconds() / _profile.calls);
    }
    double get_mean_seconds() {
        return (get_accumulated_seconds() / _profile.calls);
    }
    double get_accumulated() {
        return (_profile.accumulated);
    }
    double get_accumulated_useconds() {
        return (_profile.accumulated * 1.0e-3);
    }
    double get_accumulated_seconds() {
        return (_profile.accumulated * 1.0e-9);
    }
    double * get_papi_metrics() { return (_profile.papi_metrics); }
    double get_minimum() {
        return (_profile.minimum);
    }
    double get_maximum() {
        return (_profile.maximum);
    }
    int get_times_reset() { return (_profile.times_reset); }
    double get_variance() {
        double mean = get_mean();
        double variance = ((_profile.sum_squares / _profile.calls) - (mean * mean));
        return variance >= 0.0 ? variance : 0.0;
    }
    double get_sum_squares() {
        return _profile.sum_squares;
    }
    double get_stddev() { return sqrt(get_variance()); }
    double get_allocations() { return _profile.allocations; }
    double get_frees() { return _profile.frees; }
    double get_bytes_allocated() { return _profile.bytes_allocated; }
    double get_bytes_freed() { return _profile.bytes_freed; }
    apex_profile_type get_type() { return _profile.type; }
    apex_profile * get_profile() { return &_profile; };

};


}

