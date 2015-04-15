#include <iostream>
#include <array>
#include <algorithm>
#include "apex.hpp"
#if defined (_OPENMP)
#include "omp.h"
#else 
#define omp_get_max_threads() 1
#endif

#define NUM_CELLS 100000
#define BLOCK_SIZE NUM_CELLS/100
#define NUM_ITERATIONS 1000
#define UPDATE_INTERVAL NUM_ITERATIONS/100

long num_cells = NUM_CELLS;
long block_size = BLOCK_SIZE;
long active_threads = omp_get_max_threads();
long num_iterations = NUM_ITERATIONS;
long update_interval = UPDATE_INTERVAL;
apex_event_type my_custom_event = APEX_CUSTOM_EVENT;

void parse_arguments(int argc, char ** argv) {
    if (argc > 1 && argc < 7) {
        std::cout << "Usage: " << argv[0] << " -c <num_cells> -b <block_size> -i <iterations>" << std::endl;
        exit(0);
    }
    for (int i = 1 ; i < argc ; i+=2) {
        if (strncmp(argv[i], "-c", 2) == 0) {
            std::cout << "num_cells: " << argv[i+1] << std::endl;
            num_cells = atoi(argv[i+1]);
        }
        if (strncmp(argv[i], "-b", 2) == 0) {
            std::cout << "block_size: " << argv[i+1] << std::endl;
            block_size = atoi(argv[i+1]);
        }
        if (strncmp(argv[i], "-i", 2) == 0) {
            std::cout << "iterations: " << argv[i+1] << std::endl;
            num_iterations = atoi(argv[i+1]);
        }
    }
    update_interval = num_iterations / 100;
}

std::vector<double> * initialize(bool zeroes) {
    std::vector<double> * my_array = new std::vector<double>(num_cells);
    double value = 0.0;
    for ( auto it = my_array->begin(); it != my_array->end(); ++it ) {
        if (zeroes) {
            *it = 0.0;
        } else {
            //*it = value;
            *it = ((double) rand() / (RAND_MAX));;
        }
        value = value + 1.0;
    }
    return my_array;
}

void dump_array(std::vector<double> * my_array) {
    std::cout << "my_array contains: [";
    for ( auto it = my_array->begin(); it != my_array->end(); ++it ) {
        std::cout << ',' << *it;
    }
    std::cout << ']' << std::endl;
    return;
}

inline void solve_cell(std::vector<double> & in_array, std::vector<double> & out_array, long index) {
    if (index == 0) {
        out_array[index] = (in_array[index] + in_array[index+1]) / 2.0;
    } else if (index == num_cells - 1) {
        out_array[index] = (in_array[index-1] + in_array[index]) / 2.0;
    } else {
        out_array[index] = (in_array[index-1] + in_array[index] + in_array[index+1]) / 3.0;
    }
}

void solve(std::vector<double> & in_array, std::vector<double> & out_array,
                long start_index, long end_index) {
    apex::profiler* p = apex::start((apex_function_address)solve);
    long index = 0;
    end_index = std::min(end_index, num_cells);
    for ( index = start_index ; index < end_index ; index++) {
        solve_cell(in_array, out_array, index);
    }
    apex::stop(p);
}

void solve_iteration(std::vector<double> * in_array, std::vector<double> * out_array) {
    apex::profiler* p = apex::start((apex_function_address)solve_iteration);
#pragma omp parallel num_threads(active_threads)
    {
#pragma omp single
        for (long j = 0; j < num_cells ; j += block_size) {
#pragma omp task
            solve(*in_array,*out_array,j,j+block_size);
        }
// #pragma omp taskwait
    }
    apex::stop(p);
}

void report_stats(void) {
    double num_blocks = (double)num_cells / (double)block_size;
    double blocks_per_thread = num_blocks / (double)active_threads;
    std::cout << "number of blocks: " << num_blocks;
    std::cout << ", blocks per thread: " << blocks_per_thread << std::endl;
}

int main (int argc, char ** argv) {
    apex::init(argc, argv, "openmp test");
    parse_arguments(argc, argv);
    apex::set_node_id(0);

#ifdef APEX_HAVE_ACTIVEHARMONY
    int num_inputs = 2;
    long * inputs[2] = {0L,0L};
    long mins[2] = {1,1};    // all minimums are 1
    long maxs[2] = {0,0};    // we'll set these later
    long steps[2] = {1,1};   // all step sizes are 1
    inputs[0] = &active_threads;
    inputs[1] = &block_size;
    maxs[0] = active_threads;
    maxs[1] = num_cells/omp_get_max_threads();
    std::cout <<"Tuning Parameters:" << std::endl;
    std::cout <<"\tmins[0]: " << mins[0] << ", maxs[0]: " << maxs[0] << ", steps[0]: " << steps[0] << std::endl;
    my_custom_event = apex::register_custom_event("Perform Re-block");
    apex::setup_general_tuning((apex_function_address)solve_iteration,
                    APEX_MINIMIZE_ACCUMULATED, my_custom_event, num_inputs,
                    inputs, mins, maxs, steps);
#endif
    std::cout <<"Running 1D stencil test..." << std::endl;

    std::vector<double> * prev = initialize(false);
    std::vector<double> * next = initialize(true);
    std::vector<double> * tmp = prev;
    double prev_accumulated = 0.0;
    for (int i = 0 ; i < num_iterations ; i++) {
        solve_iteration(prev, next);
        //dump_array(next);
        tmp = prev;
        prev = next;
        next = tmp;
        if (i % update_interval == 0 && i > 0) {
            apex_profile * p = apex::get_profile((apex_function_address)solve_iteration);
            if (p != nullptr) {
                double next_accumulated = p->accumulated - prev_accumulated;
                prev_accumulated = p->accumulated;
                std::cout << "Iteration: " << i << " accumulated: " << next_accumulated << std::endl;
            }
            apex::custom_event(my_custom_event, NULL);
            std::cout << "New thread count: " << active_threads;
            std::cout << ", New block size: " << block_size << std::endl;
        }
    }
    //dump_array(tmp);
    report_stats();
    delete(prev);
    delete(next);
    std::cout << "done." << std::endl;
    apex::finalize();
}
