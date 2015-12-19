#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <iostream>
#include <sstream>
#include <math.h>
#include "apex_options.hpp"
#include "apex_types.h"
#include <chrono>

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

template<std::intmax_t clock_freq>
struct rdtsc_clock {
    typedef unsigned long long rep;
    typedef std::ratio<1, clock_freq> period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<rdtsc_clock> time_point;
    static const bool is_steady = true;
    static time_point now() noexcept {
        unsigned lo, hi;
        asm volatile("rdtsc" : "=a" (lo), "=d" (hi));
        return time_point(duration(static_cast<rep>(hi) << 32 | lo));
    }
};

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
        start(MYCLOCK::now()), 
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
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
        start(MYCLOCK::now()), 
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
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
        end = MYCLOCK::now();
        stopped = true;
    };
    void stop() {
        end = MYCLOCK::now();
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
                asm volatile("");
            }
            auto t1a = MYCLOCK::now();
            auto t1b = std::chrono::CLOCK_TYPE::now();
            // Get the clock ticks per time period
            //std::cout << CycleA(t1a-t0a).count() << " 1MHz ticks seen." << std::endl;
            //std::cout << std::chrono::duration_cast<std::chrono::seconds>(CycleB(t1b-t0b)).count() << " Seconds? seen." << std::endl;
            ticks_per_period = CycleB(t1b-t0b)/CycleA(t1a-t0a);
            if (apex_options::use_screen_output()) {
                std::cout << "CPU is " << (1.0/ticks_per_period) << " Hz." << std::endl;
            }
        }
        return ticks_per_period;
#endif
    }
};

}

#endif
