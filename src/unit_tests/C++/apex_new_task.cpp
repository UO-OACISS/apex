#include <thread>
#include <future>
#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include <atomic>

#define FIB_RESULTS_PRE 41
int fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

std::atomic<uint64_t> task_id(-1);

int fib (int in, std::shared_ptr< apex::task_wrapper > this_task) {
    apex::start(this_task);
    if (in == 0) {
        return 0;
    }
    else if (in == 1) {
        return 1;
    }
    int a = in-1;
	// called from the parent, not when the child is spawned!
    auto task_1 = apex::new_task((apex_function_address)&fib, ++task_id);
    auto future_a = std::async(std::launch::async, fib, a, task_1);

    int b = in-2;
	// called from the parent, not when the child is spawned!
    auto task_2 = apex::new_task((apex_function_address)&fib, ++task_id);
    auto future_b = std::async(std::launch::async, fib, b, task_2);

    int result_a = future_a.get();
    int result_b = future_b.get();
    apex::stop(this_task);
    return (result_a + result_b);
}

int main(int argc, char *argv[]) {
    apex::init("apex_new_task_cpp unit test", 0, 1);
    apex::scoped_timer foo(__func__);
#if defined(APEX_HAVE_TAU) || (APEX_HAVE_OTF2)
    int i = 5;
#else
    int i = 10;
#endif

    if (argc != 2) {
        std::cerr << "usage: pthreads <integer value>" << std::endl;
        std::cerr << "Using default value of 10" << std::endl;
    } else {
        i = atol(argv[1]);
    }

    if (i < 1) {
        std::cerr << i << " must be>= 1" << std::endl;
        return -1;
    }

	// called from the parent, not when the child is spawned!
    auto task = apex::new_task((apex_function_address)&fib, ++task_id);
    auto future = std::async(std::launch::async, fib, i, task);
    int result = future.get();
    std::cout << "fib of " << i << " is " << result << " (valid value: " << fib_results[i] << ")" << std::endl;
	foo.stop();
    apex::finalize();
    return 0;
}

