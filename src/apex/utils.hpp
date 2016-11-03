//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

namespace apex {

bool starts_with(const std::string& input, const std::string& match);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

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
        simple_timer() : start(std::chrono::high_resolution_clock::now()) {}
        ~simple_timer() {
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "simple time: " << time_span.count() * nanoseconds << "ns" << std::endl;
        }
};

#if !defined(_MSC_VER)
inline unsigned int my_hardware_concurrency()
{
    return sysconf(_SC_NPROCESSORS_ONLN);
}
#endif

inline unsigned int hardware_concurrency()
{
#if defined(_MSC_VER)
    return std::thread::hardware_concurrency();
#else
    unsigned int cores = std::thread::hardware_concurrency();
    return cores ? cores : my_hardware_concurrency();
#endif
}

std::string demangle(const std::string& timer_name);

void set_thread_affinity(void);

void remove_path(const char * pathname);

};

