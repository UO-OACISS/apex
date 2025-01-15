/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

// forward declaration
namespace apex {
class profiler;
}

#include <iostream>
#include <sstream>
#include <math.h>
#include <memory>
#include <map>
#include "apex_options.hpp"
#include "apex_types.h"
#include "apex_assert.h"
#include "apex_clock.hpp"
#include "task_wrapper.hpp"

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

class profiler : public std::enable_shared_from_this<profiler> {
private:
    task_identifier * task_id; // for counters, timers
public:
    std::shared_ptr<task_wrapper> tt_ptr;     // for timers
    uint64_t start_ns;
    uint64_t end_ns;
#if APEX_HAVE_PAPI
    long long papi_start_values[8];
    long long papi_stop_values[8];
#endif
    double allocations;
    double frees;
    double bytes_allocated;
    double bytes_freed;
    double value;
    double children_value;
    uint64_t guid;
    bool is_counter;
    bool is_resume; // for yield or resume
    reset_type is_reset;
    bool stopped;
    // needed for correct Hatchet output
    uint64_t thread_id; // saved at timer start
    std::map<std::string, double> metric_map;
    task_identifier * get_task_id(void) {
        return task_id;
    }
    void set_task_id(task_identifier * tid) {
        task_id = tid;
    }
    // this constructor is for regular timers
    profiler(std::shared_ptr<task_wrapper> &task,
             bool resume = false,
             reset_type reset = reset_type::NONE) :
        task_id(task->get_task_id()),
        tt_ptr(task),
        start_ns(our_clock::now_ns()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        allocations(0), frees(0), bytes_allocated(0), bytes_freed(0),
        value(0.0),
        children_value(0.0),
        guid(task->guid),
        is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false),
        thread_id(task->thread_id) {
            //printf("constructor! %p\n", this); fflush(stdout);
            task->prof = this;
            task->assign_start_ns(start_ns);
        }
    // this constructor is for resetting profile values
    profiler(task_identifier * id,
             bool resume = false,
             reset_type reset = reset_type::NONE) :
        task_id(id),
        tt_ptr(nullptr),
        start_ns(our_clock::now_ns()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        allocations(0), frees(0), bytes_allocated(0), bytes_freed(0),
        value(0.0),
        children_value(0.0),
        guid(0),
        is_counter(false),
        is_resume(resume),
        is_reset(reset), stopped(false), thread_id(0) { };
    // this constructor is for counters
    profiler(task_identifier * id, double value_) :
        task_id(id),
        tt_ptr(nullptr),
        start_ns(our_clock::now_ns()),
#if APEX_HAVE_PAPI
        papi_start_values{0,0,0,0,0,0,0,0},
        papi_stop_values{0,0,0,0,0,0,0,0},
#endif
        allocations(0), frees(0), bytes_allocated(0), bytes_freed(0),
        value(value_),
        children_value(0.0),
        is_counter(true),
        is_resume(false),
        is_reset(reset_type::NONE), stopped(true) { };
    //copy constructor
    profiler(const profiler& in) :
        std::enable_shared_from_this<profiler>(in),
        task_id(in.task_id),
        tt_ptr(in.tt_ptr),
        start_ns(in.start_ns),
        end_ns(in.end_ns),
        allocations(in.allocations),
        frees(in.frees),
        bytes_allocated(in.bytes_allocated),
        bytes_freed(in.bytes_freed),
        value(in.value),
        children_value(in.children_value),
        guid(in.guid),
        is_counter(in.is_counter),
        is_resume(in.is_resume), // for yield or resume
        is_reset(in.is_reset),
        stopped(in.stopped),
        thread_id(in.thread_id)
    {
        //printf("COPY! %p -> %p\n", &in, this); fflush(stdout);
#if APEX_HAVE_PAPI
        for (int i = 0 ; i < 8 ; i++) {
            papi_start_values[i] = in.papi_start_values[i];
            papi_stop_values[i] = in.papi_stop_values[i];
        }
#endif
    }
    ~profiler(void) { /* not much to do here. */
        //printf("destructor! %p\n", this); fflush(stdout);
    };
    // for "yield" support
    void set_start(uint64_t timestamp) {
        start_ns = timestamp;
    }
    void set_end(uint64_t timestamp) {
        end_ns = timestamp;
    }
    void stop(bool is_resume) {
        this->is_resume = is_resume;
        end_ns = our_clock::now_ns();
        stopped = true;
    };
    void stop() {
        end_ns = our_clock::now_ns();
        stopped = true;
    };
    void restart() {
        this->is_resume = true;
        start_ns = our_clock::now_ns();
    };
    uint64_t get_start_ns() {
        return start_ns;
    }
    double get_start_us() {
        return start_ns*1.0e-3;
    }
    double get_start_ms() {
        return start_ns*1.0e-6;
    }
    uint64_t get_stop_ns() {
        return end_ns;
    }
    double get_stop_us() {
        return end_ns*1.0e-3;
    }
    double get_stop_ms() {
        return end_ns*1.0e-6;
    }
    static double now_us( void ) {
        double stamp = (double)our_clock::now_ns();
        return stamp*1.0e-3;
    }
    static double now_ms( void ) {
        double stamp = (double)our_clock::now_ns();
        return stamp*1.0e-6;
    }
    double elapsed() {
        if(is_counter) {
            return value;
        } else {
            if (!stopped) {
                end_ns = our_clock::now_ns();
            }
            return ((double)(end_ns-start_ns));
        }
    }
    double inclusive() {
        if(is_counter) { return 0.0; }
        /* The task isn't done yet. */
        if (is_resume) {
            return (0.0);
        } else {
            /* Is this an asynchronous event? if so, the task lifetime
             * will be after the profiler lifetime. So check for that. */
            double d_incl_start = (double)(
                start_ns < tt_ptr->get_start_ns() ?
                start_ns : tt_ptr->get_start_ns());
            double d_end = (double)(end_ns);
            if (!stopped) {
                d_end = (double)our_clock::now_ns();
            }
            double incl = d_end - d_incl_start;
            return (incl);
        }
    }
    double inclusive_seconds() {
        return inclusive() * 1.0e-9;
    }
    double elapsed_us() {
        return elapsed() * 1.0e-3;
    }
    double elapsed_ms() {
        return elapsed() * 1.0e-6;
    }
    double elapsed_seconds() {
        return elapsed() * 1.0e-9;
    }
    double exclusive_elapsed(void) {
        return elapsed() - children_value;
    }

    static uint64_t now_ns() {
        return our_clock::now_ns();
    }

    static profiler* get_disabled_profiler(void) {
        static profiler disabled_profiler;
        return &disabled_profiler;
    }
    // default constructor for the dummy profiler
    profiler(void) {};
    // dummy profiler to indicate that stop/yield should resume immediately

    /* this is for OTF2 tracing.
     * We want a timestamp for the start of the trace.
     * We will also need one for the end of the trace. */
    static uint64_t get_global_start(void) {
        static uint64_t global_now = our_clock::now_ns();
        return global_now;
    }
    /* this is for getting the endpoint of the trace. */
    static uint64_t get_global_end(void) {
        return our_clock::now_ns();
    }
    double normalized_timestamp(void) {
        if(is_counter) {
            return our_clock::now_ns() - get_global_start();
        } else {
            return start_ns - get_global_start();
        }
    }
    std::shared_ptr<profiler> Get() {return shared_from_this();}
};

}

