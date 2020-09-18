#pragma once

// #include "tsc_x86.h"
#include <chrono>

namespace {
    std::chrono::high_resolution_clock::time_point __start_time;
    // unsigned long long int __start_cycles;
}

void startTimer() { __start_time = std::chrono::high_resolution_clock::now(); }

// void startCycles() { __start_cycles = start_tsc(); }

double endTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - __start_time);
    return time_span.count();
}

// auto endCycles() { return stop_tsc(__start_cycles); }