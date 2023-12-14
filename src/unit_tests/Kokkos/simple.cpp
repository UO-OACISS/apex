#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <Kokkos_Core.hpp>

#ifndef EXECUTION_SPACE
#define EXECUTION_SPACE DefaultHostExecutionSpace
#endif

void go(size_t i) {
    int n = 512;
    Kokkos::View<double*> a("a",n);
    Kokkos::View<double*> b("b",n);
    Kokkos::View<double*> c("c",n);

    auto range = Kokkos::RangePolicy<Kokkos::EXECUTION_SPACE>(0,n);

    Kokkos::parallel_for(
        "initialize", range, KOKKOS_LAMBDA(size_t const i) {
            auto x = static_cast<double>(i);
            a(i) = sin(x) * sin(x);
            b(i) = cos(x) * cos(x);
        }
    );

    Kokkos::parallel_for(
        "xpy", range, KOKKOS_LAMBDA(size_t const i) {
            c(i) = a(i) + b(i);
        }
    );

    double sum = 0.0;

    Kokkos::parallel_reduce(
        "sum", range, KOKKOS_LAMBDA(size_t const i, double &lsum) {
            lsum += c(i);
        }, sum
    );

    if (i % 10 == 0) {
        std::cout << "sum = " << sum / n << std::endl;
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    std::cout << "Kokkos execution space: "
        << Kokkos::DefaultExecutionSpace::name() << std::endl;
    for (size_t i = 0 ; i < 10 ; i++) {
        go(i);
    }
    Kokkos::finalize();
}