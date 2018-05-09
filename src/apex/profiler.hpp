//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// forward declaration
namespace apex {
class profiler;
};

#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"
#include <chrono>
#include "task_wrapper.hpp"
#if defined(APEX_HAVE_HPX)
#include <hpx/util/hardware/timestamp.hpp>
#endif

#ifdef __INTEL_COMPILER
#define CLOCK_TYPE high_resolution_clock
#else
#define CLOCK_TYPE steady_clock
#endif

namespace apex {

enum struct reset_type {
    NONE,    // not a reset event
    CURRENT, // reset the specified counter
    ALL     // reset all counters
};

class disabled_profiler_exception : public std::exception {
    virtual const char* what() const throw() {
      return "Disabled profiler.";
    }
};

#ifndef APEX_USE_CLOCK_TIMESTAMP
template<std::intmax_t clock_freq>
struct rdtsc_clock {
    typedef unsigned long long rep;
    typedef std::ratio<1, clock_freq> period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<rdtsc_clock> time_point;
    static const bool is_steady = true;
    static time_point now() noexcept {
#if defined(APEX_HAVE_HPX)
        return time_point(duration(hpx::util::hardware::timestamp()));
#else
        unsigned lo, hi;
        asm volatile("rdtsc" : "=a" (lo), "=d" (hi));
        return time_point(duration(static_cast<rep>(hi) << 32 | lo));
#endif
    }
};
#endif

#ifdef APEX_USE_CLOCK_TIMESTAMP
#define MYCLOCK std::chrono::CLOCK_TYPE
#else
typedef rdtsc_clock<1> OneHzClock;
#define MYCLOCK OneHzClock
#endif

class profiler {
public:
    MYCLOCK::time_point start;
    MYCLOCK::time_point end;
#if APEX_HAVE_PAPI
    long long papi_start_values[8];
    long long papi_stop_values[8];
#endif
    double value;
    double children_value;
    //apex_function_address action_address;
    //std::string * timer_name;
    //bool have_name;
	task_wrapper * tt_ptr;     // for timers
	task_identifier * task_id; // for counters, timers
    uint64_t guid;
    bool is_counter;
    bool is_resume; // for yield or resume
    reset_type is_reset;
    bool stopped;
    profiler(task_wrapper * task,
             bool resume = false,
             reset_type reset = reset_type::NONE) :
        start(MYCLOCK::now()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        value(0.0),
        children_value(0.0),
		tt_ptr(task),
		task_id(tt_ptr->get_task_id()),
        guid(0),
        is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false) { };
    // this constructor is for resetting profile values
    profiler(task_identifier * id,
             bool resume = false,
             reset_type reset = reset_type::NONE) :
        start(MYCLOCK::now()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        value(0.0),
        children_value(0.0),
		tt_ptr(nullptr),
		task_id(id),
        guid(0),
        is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false) { };
    profiler(task_identifier * id, double value_) :
        start(MYCLOCK::now()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        value(value_),
        children_value(0.0),
		tt_ptr(nullptr),
		task_id(id),
        is_counter(true),
        is_resume(false),
        is_reset(reset_type::NONE), stopped(true) { };
    //copy constructor
    profiler(const profiler& in) : start(in.start), end(in.end) {
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < 8 ; i++) {
            papi_start_values[i] = in.papi_start_values[i];
            papi_stop_values[i] = in.papi_stop_values[i];
        }
#endif
    value = in.value;
    children_value = in.children_value;
	tt_ptr = in.tt_ptr;
	task_id = in.task_id;
    is_counter = in.is_counter;
    is_resume = in.is_resume; // for yield or resume
    is_reset = in.is_reset;
    stopped = in.stopped;
    }
    ~profiler(void) { /* not much to do here. */ };
    // for "yield" support
    void stop(bool is_resume) {
        this->is_resume = is_resume;
        end = MYCLOCK::now();
        stopped = true;
    };
    void stop() {
        end = MYCLOCK::now();
        stopped = true;
    };
    void restart() {
        this->is_resume = true;
        start = MYCLOCK::now();
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

    /* This function returns 1/X, where "X" is the MHz rating of the CPU. */
    static double get_cpu_mhz () {
#ifdef APEX_USE_CLOCK_TIMESTAMP
        return 1.0;
#else
        static double ticks_per_period = 0.0;
        if (ticks_per_period == 0.0) {
            typedef std::chrono::duration<double, typename MYCLOCK::period> CycleA;
            typedef std::chrono::duration<double, typename std::chrono::CLOCK_TYPE::period> CycleB;
            const int N = 100000000;
            auto t0a = MYCLOCK::now();
            auto t0b = std::chrono::CLOCK_TYPE::now();
            for (int j = 0; j < N; ++j) {
#if !defined(_MSC_VER)
                asm volatile("");
#endif
            }
            auto t1a = MYCLOCK::now();
            auto t1b = std::chrono::CLOCK_TYPE::now();
            // Get the clock ticks per time period
            //std::cout << CycleA(t1a-t0a).count() << " 1MHz ticks seen." << std::endl;
            //std::cout << std::chrono::duration_cast<std::chrono::seconds>(CycleB(t1b-t0b)).count() << " Seconds? seen." << std::endl;
            ticks_per_period = CycleB(t1b-t0b)/CycleA(t1a-t0a);
            /*
            if (apex_options::use_screen_output()) {
                std::cout << "CPU is " << (1.0/ticks_per_period) << " Hz." << std::endl;
            }
            */
        }
        return ticks_per_period;
#endif
    }

    /* this is for OTF2 tracing.
     * We want a timestamp for the start of the trace.
     * We will also need one for the end of the trace. */
    static MYCLOCK::time_point get_global_start(void) {
        static MYCLOCK::time_point global_now = MYCLOCK::now();
        return global_now;
    }
    /* this is for getting the endpoint of the trace. */
    static MYCLOCK::time_point get_global_end(void) {
        return MYCLOCK::now();
    }
    static uint64_t time_point_to_nanoseconds(MYCLOCK::time_point tp) {
        auto value = tp.time_since_epoch();
        uint64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
        return duration;
    }
    double normalized_timestamp(void) {
        if(is_counter) {
            return value;
        } else {
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(start - get_global_start());
            return time_span.count()*get_cpu_mhz();
        }
    }
};

}

