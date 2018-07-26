#include <omp.h>
#include <stdio.h>
#include "apex.h"

void a() {
    printf("in function a\n");
}
void b() {
    printf("in function b\n");
}
void c() {
    printf("in function c\n");
}

int main (void) {
 	apex_init(__func__, 0, 1);
    apex_set_use_screen_output(1);
#pragma omp parallel sections
    {
    #pragma omp section
        a();
    #pragma omp section
        b();
    #pragma omp section
        c();
    }
    apex_finalize();
}