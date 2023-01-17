#include <unistd.h>
#include <stdio.h>
#include <thread>
#include <string>
#include <cinttypes>
#include "apex_api.hpp"

#define MAX_OUTER 50
#define MAX_INNER 50
#define MAX_THREADS 8

uint64_t func(uint64_t i) {
    char name[128];
    snprintf(name, 128, "func %llu", static_cast<unsigned long long>(i));
    apex::profiler* p = apex::start(std::string(name));
    uint64_t j = i * i;
    apex::stop(p);
    return j;
}

uint64_t foo(uint64_t i) {
    uint64_t j=0;
    apex::register_thread(__func__);
    apex::profiler* p = apex::start((apex_function_address)(&foo));
    for (uint64_t x = 0 ; x < MAX_OUTER ; x++) {
        for (uint64_t y = 0 ; y < MAX_INNER ; y++) {
            j += func(x) * func(y) + i;
        }
    }
    apex::stop(p);
    return j;
}

// no timer!
uint64_t bar(uint64_t i) {
    // ask for a thread instance, as a test.
    //
    // create a task, but don't start a timer.
    apex::new_task((apex_function_address)&bar);
    uint64_t j=0;
    for (uint64_t x = 0 ; x < MAX_OUTER ; x++) {
        for (uint64_t y = 0 ; y < MAX_INNER ; y++) {
            j += (x*x) * (y*y) + i;
        }
    }
    return j;
}

int main (int argc, char** argv) {
    APEX_UNUSED(argc);
    APEX_UNUSED(argv);
    apex::init("apex_start unit test", 0, 1);
    apex::profiler* p = apex::start(__func__);
    uint64_t i = 0;
    std::thread threads[MAX_THREADS];
    for (i = 0 ; i < MAX_THREADS ; i++) {
        //j += foo(i);
        if (i % 2 == 0) {
            // create a worker thread
            threads[i] = std::thread(foo,i);
        } else {
            // create a non-worker thread
            threads[i] = std::thread(bar,i);
        }
    }
    for (i = 0 ; i < MAX_THREADS ; i++) {
        threads[i].join();
    }
    apex::stop(p);
    apex::finalize();
    apex::cleanup();
    return 0;
}

