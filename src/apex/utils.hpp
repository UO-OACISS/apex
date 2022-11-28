/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <string>
#include <chrono>
#include <thread>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif
#include <atomic>

#ifdef DEBUG
#define DEBUG_PRINT(...) if (::apex::apex_options::use_verbose()) { \
fprintf( stderr, __VA_ARGS__ ); fflush(stderr); }
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

#include "apex_types.h"

namespace apex {

bool starts_with(const std::string& input, const std::string& match);

std::vector<std::string> &split(const std::string &s, char delim,
    std::vector<std::string> &elems);

// trim from left
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from right
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from left & right
inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v")
{
    return ltrim(rtrim(s, t), t);
}

// remove substrings
inline void eraseSubStr(std::string &mainStr, const std::string toErase) {
    // Search for the substring in string
    size_t pos = mainStr.find(toErase);
    if (pos != std::string::npos) {
        // If found then erase it from string
        mainStr.erase(pos, toErase.length());
    }
}

// copying versions

inline std::string ltrim_copy(std::string s, const char* t = " \t\n\r\f\v")
{
    return ltrim(s, t);
}

inline std::string rtrim_copy(std::string s, const char* t = " \t\n\r\f\v")
{
    return rtrim(s, t);
}

inline std::string trim_copy(std::string s, const char* t = " \t\n\r\f\v")
{
    return trim(s, t);
}

class simple_timer {
        const double nanoseconds = 1.0e9;
    public:
        std::chrono::high_resolution_clock::time_point start;
        std::string _name;
        simple_timer() :
            start(std::chrono::high_resolution_clock::now()), _name("") {}
        simple_timer(std::string name) :
            start(std::chrono::high_resolution_clock::now()), _name(name) {}
        ~simple_timer() {
            std::chrono::duration<double> time_span =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - start);
            std::cout << _name << " simple time: " <<
                time_span.count() * nanoseconds << "ns" << std::endl;
        }
};

class reference_counter {
    public:
        static std::atomic<uint64_t> task_wrappers;
        static std::atomic<uint64_t> null_task_wrappers;

        static std::atomic<uint64_t> starts;
        static std::atomic<uint64_t> disabled_starts;
        static std::atomic<uint64_t> apex_internal_starts;
        static std::atomic<uint64_t> hpx_shutdown_starts;
        static std::atomic<uint64_t> hpx_timer_starts;
        static std::atomic<uint64_t> suspended_starts;
        static std::atomic<uint64_t> failed_starts;
        static std::atomic<uint64_t> starts_after_finalize;

        static std::atomic<uint64_t> resumes;
        static std::atomic<uint64_t> disabled_resumes;
        static std::atomic<uint64_t> apex_internal_resumes;
        static std::atomic<uint64_t> hpx_shutdown_resumes;
        static std::atomic<uint64_t> hpx_timer_resumes;
        static std::atomic<uint64_t> suspended_resumes;
        static std::atomic<uint64_t> failed_resumes;
        static std::atomic<uint64_t> resumes_after_finalize;

        static std::atomic<uint64_t> yields;
        static std::atomic<uint64_t> disabled_yields;
        static std::atomic<uint64_t> null_yields;
        static std::atomic<uint64_t> double_yields;
        static std::atomic<uint64_t> yields_after_finalize;
        static std::atomic<uint64_t> apex_internal_yields;

        static std::atomic<uint64_t> stops;
        static std::atomic<uint64_t> disabled_stops;
        static std::atomic<uint64_t> null_stops;
        static std::atomic<uint64_t> double_stops;
        static std::atomic<uint64_t> stops_after_finalize;
        static std::atomic<uint64_t> apex_internal_stops;
        static std::atomic<uint64_t> exit_stops;
        static void report_stats(void);
};

#if defined(APEX_DEBUG)
#define APEX_UTIL_REF_COUNT_TASK_WRAPPER         reference_counter::task_wrappers++;
#define APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER    reference_counter::null_task_wrappers++;

#define APEX_UTIL_REF_COUNT_START                reference_counter::starts++;
#define APEX_UTIL_REF_COUNT_DISABLED_START       reference_counter::disabled_starts++;
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_START  \
    reference_counter::apex_internal_starts++;
#define APEX_UTIL_REF_COUNT_HPX_SHUTDOWN_START   \
    reference_counter::hpx_shutdown_starts++;
#define APEX_UTIL_REF_COUNT_HPX_TIMER_START      reference_counter::hpx_timer_starts++;
#define APEX_UTIL_REF_COUNT_SUSPENDED_START      reference_counter::suspended_starts++;
#define APEX_UTIL_REF_COUNT_FAILED_START         reference_counter::failed_starts++;
#define APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE \
    reference_counter::starts_after_finalize++;

#define APEX_UTIL_REF_COUNT_RESUME               reference_counter::resumes++;
#define APEX_UTIL_REF_COUNT_DISABLED_RESUME      reference_counter::disabled_resumes++;
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME \
    reference_counter::apex_internal_resumes++;
#define APEX_UTIL_REF_COUNT_HPX_SHUTDOWN_RESUME  \
    reference_counter::hpx_shutdown_resumes++;
#define APEX_UTIL_REF_COUNT_HPX_TIMER_RESUME     reference_counter::hpx_timer_resumes++;
#define APEX_UTIL_REF_COUNT_SUSPENDED_RESUME     reference_counter::suspended_resumes++;
#define APEX_UTIL_REF_COUNT_FAILED_RESUME        reference_counter::failed_resumes++;
#define APEX_UTIL_REF_COUNT_RESUME_AFTER_FINALIZE \
    reference_counter::resumes_after_finalize++;

#define APEX_UTIL_REF_COUNT_YIELD                reference_counter::yields++;
#define APEX_UTIL_REF_COUNT_DISABLED_YIELD       reference_counter::disabled_yields++;
#define APEX_UTIL_REF_COUNT_NULL_YIELD           reference_counter::null_yields++;
#define APEX_UTIL_REF_COUNT_DOUBLE_YIELD         reference_counter::double_yields++;
#define APEX_UTIL_REF_COUNT_YIELD_AFTER_FINALIZE \
    reference_counter::yields_after_finalize++;
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD  \
    reference_counter::apex_internal_yields++;

#define APEX_UTIL_REF_COUNT_STOP                 reference_counter::stops++;
#define APEX_UTIL_REF_COUNT_DISABLED_STOP        reference_counter::disabled_stops++;
#define APEX_UTIL_REF_COUNT_NULL_STOP            reference_counter::null_stops++;
#define APEX_UTIL_REF_COUNT_DOUBLE_STOP          reference_counter::double_stops++;
#define APEX_UTIL_REF_COUNT_EXIT_STOP            reference_counter::exit_stops++;
#define APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE  \
    reference_counter::stops_after_finalize++;
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP   \
    reference_counter::apex_internal_stops++;
#define APEX_UTIL_REPORT_STATS                   reference_counter::report_stats();
#else
#define APEX_UTIL_REF_COUNT_TASK_WRAPPER
#define APEX_UTIL_REF_COUNT_NULL_TASK_WRAPPER

#define APEX_UTIL_REF_COUNT_START
#define APEX_UTIL_REF_COUNT_DISABLED_START
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_START
#define APEX_UTIL_REF_COUNT_HPX_SHUTDOWN_START
#define APEX_UTIL_REF_COUNT_HPX_TIMER_START
#define APEX_UTIL_REF_COUNT_SUSPENDED_START
#define APEX_UTIL_REF_COUNT_FAILED_START
#define APEX_UTIL_REF_COUNT_START_AFTER_FINALIZE

#define APEX_UTIL_REF_COUNT_RESUME
#define APEX_UTIL_REF_COUNT_DISABLED_RESUME
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_RESUME
#define APEX_UTIL_REF_COUNT_HPX_SHUTDOWN_RESUME
#define APEX_UTIL_REF_COUNT_HPX_TIMER_RESUME
#define APEX_UTIL_REF_COUNT_SUSPENDED_RESUME
#define APEX_UTIL_REF_COUNT_FAILED_RESUME
#define APEX_UTIL_REF_COUNT_RESUME_AFTER_FINALIZE

#define APEX_UTIL_REF_COUNT_YIELD
#define APEX_UTIL_REF_COUNT_DISABLED_YIELD
#define APEX_UTIL_REF_COUNT_NULL_YIELD
#define APEX_UTIL_REF_COUNT_DOUBLE_YIELD
#define APEX_UTIL_REF_COUNT_YIELD_AFTER_FINALIZE
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_YIELD

#define APEX_UTIL_REF_COUNT_STOP
#define APEX_UTIL_REF_COUNT_DISABLED_STOP
#define APEX_UTIL_REF_COUNT_NULL_STOP
#define APEX_UTIL_REF_COUNT_DOUBLE_STOP
#define APEX_UTIL_REF_COUNT_EXIT_STOP
#define APEX_UTIL_REF_COUNT_STOP_AFTER_FINALIZE
#define APEX_UTIL_REF_COUNT_APEX_INTERNAL_STOP
#define APEX_UTIL_REPORT_STATS
#endif

inline unsigned int my_hardware_concurrency()
{
#if defined(_MSC_VER)
    return std::thread::hardware_concurrency();
#else
    unsigned int cores = std::thread::hardware_concurrency();
    return cores ? cores : sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

inline unsigned int hardware_concurrency()
{
    static unsigned int hwc = my_hardware_concurrency();
    return(hwc);
}

std::string demangle(const std::string& timer_name);

void set_thread_affinity(void);
void set_thread_affinity(int core);

void remove_path(const char * pathname);
uint32_t simple_reverse(uint32_t x);

inline char filesystem_separator()
{
#if defined _WIN32 || defined __CYGWIN__
    return '\\';
#else
    return '/';
#endif
}

uint64_t test_for_MPI_comm_rank(uint64_t commrank);
uint64_t test_for_MPI_comm_size(uint64_t commsize);
void handle_delayed_start(void);

std::string activity_to_string(apex_async_activity_t activity);

class node_color {
public:
    double red;
    double green;
    double blue;
    node_color() : red(1.0), green(1.0), blue(1.0) {}
    int convert(double in) { return (int)(in * 255.0); }
} ;

node_color * get_node_color_visible(double v, double vmin, double vmax);
node_color * get_node_color(double v,double vmin,double vmax);

/* Defined in memory_wrapper_shudown.cpp */
void enable_memory_wrapper(void);
void disable_memory_wrapper(void);

#include <sys/syscall.h>

class in_apex {
    public:
        static size_t& get();
        in_apex();
        ~in_apex();
};

void rank0_print(const char * fmt, ...);

}

template <typename I>
std::string hex2str(I w, size_t hex_len = sizeof(I)<<1) {
    static const char* digits = "0123456789ABCDEF";
    std::string rc(hex_len,'0');
    for (size_t i=0, j=(hex_len-1)*4 ; i<hex_len; ++i,j-=4)
        rc[i] = digits[(w>>j) & 0x0f];
    return rc;
}


