#include "apex_api.hpp"
#include <unistd.h>
#include <stdio.h>
#include <thread>
#include <string>

#define MAX_OUTER 500
#define MAX_INNER 500
#define MAX_THREADS 8

int func(int i) {
    char name[128];
    sprintf(name, "func %d", i);
    apex::profiler* p = apex::start(std::string(name));
    int j = i * i;
    apex::stop(p);
    return j;
}

int foo(int i) {
    int j=0;
    apex::profiler* p = apex::start((apex_function_address)(&foo));
    for (int x = 0 ; x < MAX_OUTER ; x++) {
        for (int y = 0 ; y < MAX_INNER ; y++) {
            j += func(x) * func(y) + i;
        }
    }
    apex::stop(p);
    return j;
}

int main (int argc, char** argv) {
    apex::init(argc, argv, "apex_start unit test");
    apex::profiler* p = apex::start((apex_function_address)&main);
    int i = 0;
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

