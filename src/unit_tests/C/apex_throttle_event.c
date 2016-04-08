#include "apex.h"
#include <unistd.h>
#include <stdio.h>

#define MAX_OUTER 500
#define MAX_INNER 500

int func(int i) {
    char name[128];
    sprintf(name, "func %d", i);
    apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, name);
    int j = i * i;
    apex_stop(profiler);
    return j;
}

uintptr_t foo(uintptr_t i) {
    int j = 0;
    apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &foo);
    int x,y;
    for (x = 0 ; x < MAX_OUTER ; x++) {
        for (y = 0 ; y < MAX_INNER ; y++) {
            j += func(x) * func(y) + i;
        }
    }
    apex_stop(profiler);
    return j;
}

int main (int argc, char** argv) {
    apex_init_args(argc, argv, "apex_start unit test");
    apex_profiler_handle profiler = apex_start(APEX_FUNCTION_ADDRESS, &main);
    int i,j = 0;
    for (i = 0 ; i < 3 ; i++) {
        j += foo(i);
    }
    apex_stop(profiler);
    apex_finalize();
    apex_cleanup();
    return 0;
}

