#include <unistd.h>
#include <stdlib.h>
#include "apex_api.hpp"

void boo (void) {
    apex::profiler* p = apex::start((apex_function_address)&boo);
    exit(0);
    apex::stop(p);
}

void bar (void) {
    apex::profiler* p = apex::start((apex_function_address)&bar);
    boo();
    apex::stop(p);
}

void foo (void) {
    apex::profiler* p = apex::start((apex_function_address)&foo);
    bar();
    apex::stop(p);
}

int main (int argc, char** argv) {
    APEX_UNUSED(argc);
    APEX_UNUSED(argv);
    apex::apex_options::use_screen_output(true);
    apex::apex_options::top_level_os_threads(true);
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
    /* start a timer */
    apex::register_thread("main thread");
    apex::profiler* p = apex::start("main");
    /* Call our function */
    foo();
    /* stop our main timer - not reachable due to exit */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
    return 0;
}

