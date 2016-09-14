#include <thread>
#include <future>
#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include <atomic>

#define FIB_RESULTS_PRE 41
int fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

std::atomic<uint64_t> task_id(-1);

class apex_proxy {
private:
  apex::profiler * p;
public:
  apex_proxy(void * func) : p(nullptr) { 
    apex::register_thread("fib thread");
    p = apex::start((apex_function_address)func);
  }
  ~apex_proxy() { 
    if (p != nullptr) { 
      apex::stop(p);
    }
    apex::exit_thread();
  }
};

int fib (int in) {
    apex_proxy foo((void*)&fib);
    if (in == 0) {
        return 0;
    }
    else if (in == 1) {
        return 1;
    }
    int a = in-1;
	// called from the parent, not when the child is spawned!
    apex::new_task((apex_function_address)&fib, ++task_id);
    auto future_a = std::async(std::launch::async, fib, a);

    int b = in-2;
	// called from the parent, not when the child is spawned!
    apex::new_task((apex_function_address)&fib, ++task_id);
    auto future_b = std::async(std::launch::async, fib, b);

    int result_a = future_a.get();
    int result_b = future_b.get();
    return (result_a + result_b);
}

int main(int argc, char *argv[]) {
    apex::init("apex_new_task_cpp unit test");
	apex::set_node_id(0);
    apex_proxy * foo = new apex_proxy((void*)&main);
#ifdef APEX_HAVE_TAU
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
    apex::new_task((apex_function_address)&fib, ++task_id);
    auto future = std::async(std::launch::async, fib, i);
    int result = future.get();
    std::cout << "fib of " << i << " is " << result << " (valid value: " << fib_results[i] << ")" << std::endl;
	delete(foo);
    apex::finalize();
    return 0;
}

