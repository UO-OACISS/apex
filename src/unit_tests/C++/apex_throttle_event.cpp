#include "apex_api.hpp"
#include <unistd.h>
#include <stdio.h>
#include <thread>
#include <string>

#define MAX_OUTER 500
#define MAX_INNER 500
#define MAX_THREADS 8

uint64_t func(uint64_t i) {
    char name[128];
    sprintf(name, "func %lu", i);
    apex::profiler* p = apex::start(std::string(name));
    uint64_t j = i * i;
    apex::stop(p);
    return j;
}

uint64_t foo(uint64_t i) {
    uint64_t j=0;
    apex::profiler* p = apex::start((apex_function_address)(&foo));
    for (uint64_t x = 0 ; x < MAX_OUTER ; x++) {
        for (uint64_t y = 0 ; y < MAX_INNER ; y++) {
            j += func(x) * func(y) + i;
        }
    }
    apex::stop(p);
    return j;
}

int main (int argc, char** argv) {
    apex::init("apex_start unit test", 0, 1);
    apex::profiler* p = apex::start(__func__);
    uint64_t i = 0;
    std::thread threads[MAX_THREADS];
    for (i = 0 ; i < MAX_THREADS ; i++) {
        //j += foo(i);
        threads[i] = std::thread(foo,i);
    }
    for (i = 0 ; i < MAX_THREADS ; i++) {
        threads[i].join();
    }
    apex::stop(p);
    apex::finalize();
    apex::cleanup();
    return 0;
}

