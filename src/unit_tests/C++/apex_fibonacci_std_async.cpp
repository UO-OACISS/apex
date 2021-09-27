#include <thread>
#include <future>
#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"

#define FIB_RESULTS_PRE 41
int fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

int fib (int in, std::shared_ptr<apex::task_wrapper> parent) {
    apex::scoped_thread st("fib thread");
    apex::scoped_timer ast((uint64_t)&fib,
        apex::apex_options::top_level_os_threads() ? nullptr : parent);
    if (in == 0) {
        return 0;
    }
    else if (in == 1) {
        return 1;
    }
    int a = in-1;
    int b = in-2;
    auto future_a = std::async(std::launch::async, fib, a, ast.get_task_wrapper());
    auto future_b = std::async(std::launch::async, fib, b, ast.get_task_wrapper());

    //ast.yield();
    int result_a = future_a.get();
    int result_b = future_b.get();
    //ast.start();
    int result = result_a + result_b;
    return result;
}

int main(int argc, char *argv[]) {
    apex::init("apex_fibonacci_pthreads unit test", 0, 1);
#ifdef APEX_HAVE_TAU
    int i = 5;
#else
    int i = 10;
#endif

    if (argc != 2) {
        std::cerr << "usage: apex_fibonacci_std_async_cpp <integer value>" << std::endl;
        std::cerr << "Using default value of 10" << std::endl;
    } else {
        i = atol(argv[1]);
    }

    if (i < 1) {
        std::cerr << i << " must be>= 1" << std::endl;
        return -1;
    }

    std::shared_ptr<apex::task_wrapper> parent = nullptr;
    auto future = std::async(fib, i, parent);
    int result = future.get();
    std::cout << "fib of " << i << " is " << result << " (valid value: " << fib_results[i] << ")" << std::endl;
    apex::finalize();
    return 0;
}

