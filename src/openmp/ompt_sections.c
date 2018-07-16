#include <omp.h>
#include <stdio.h>

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
#pragma omp parallel sections
    {
    #pragma omp section
        a();
    #pragma omp section
        b();
    #pragma omp section
        c();
    }
}