#include <mpi.h>
#include <iostream>
#include <array>
#include <algorithm>
#include <atomic>
#if defined (_OPENMP)
#include "omp.h"
#else
#define omp_get_max_threads() 1
#endif
#include "apex_api.hpp"
#include "apex_global.h"
#include "synchronous_policy.hpp"

#define NUM_CELLS 800000
#define BLOCK_SIZE NUM_CELLS/100
#define NUM_ITERATIONS 500
#define UPDATE_INTERVAL NUM_ITERATIONS/10
#define DIVIDE_METHOD 1
#define MULTIPLY_METHOD 2

long num_cells = NUM_CELLS;
long block_size = BLOCK_SIZE;
#ifdef APEX_HAVE_TAU
long active_threads = 1;
#else
long active_threads = omp_get_max_threads();
#endif
long num_iterations = NUM_ITERATIONS;
long update_interval = UPDATE_INTERVAL;
apex_event_type my_custom_event = APEX_CUSTOM_EVENT_1;
int myrank = 0;
int num_ranks = 1;

/**
 * Parse the arguments passed into the program
 */
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
}



/**
 * Initialize the 1D stencil with zeroes or random values.
 */
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

/**
 * Dump the 1D stencil (useful for debugging)
 */
void dump_array(std::vector<double> * my_array) {
    std::cout << "my_array contains: [";
    for ( auto it = my_array->begin(); it != my_array->end(); ++it ) {
        std::cout << ',' << *it;
    }
    std::cout << ']' << std::endl;
    return;
}

/**
 * Solve one cell of the stencil, using two additions and a division
 */
inline void solve_cell(std::vector<double> & in_array, std::vector<double> & out_array, long index) {
    if (__builtin_expect(index == 0,0)) {
        out_array[index] = (in_array[index] +
                            in_array[index+1]) / 2.0;
    } else if (__builtin_expect(index == num_cells - 1,0)) {
        out_array[index] = (in_array[index-1] +
                            in_array[index]) / 2.0;
    } else {
        out_array[index] = (in_array[index-1] +
                            in_array[index] +
                            in_array[index+1]) / 3.0;
    }
}

/**
 * Solve one block of the 1D stencil, using division
 */
inline void solve(std::vector<double> & in_array, std::vector<double> & out_array,
                long start_index, long end_index) {
    apex::profiler* p = apex::start((apex_function_address)solve);
    long index = 0;
    end_index = std::min(end_index, num_cells);
    for ( index = start_index ; index < end_index ; index++) {
        solve_cell(in_array, out_array, index);
		// Create a load imbalance...
		if (myrank%2 == 0) {
          solve_cell(in_array, out_array, index);
		}
    }
    apex::stop(p);
}

/**
 * One iteration over the entire 1D stencil, using either
 * multipliation or division.
 */
void solve_iteration(std::vector<double> * in_array, std::vector<double> * out_array) {
    apex::profiler* p = apex::start((apex_function_address)solve_iteration);
#ifndef APEX_HAVE_TAU
#pragma omp parallel num_threads(active_threads)
#endif
    {
#ifndef APEX_HAVE_TAU
#pragma omp for schedule(static)
#endif
        for (long j = 0; j < num_cells ; j += block_size) {
            solve(*in_array,*out_array,j,j+block_size);
        }
    }
    apex::stop(p);
}

/**
 * Report the final tuning parameters
 */
void report_stats(void) {
    apex_profile * p = apex::get_profile((apex_function_address)solve_iteration);
    double num_blocks = (double)num_cells / (double)block_size;
    double blocks_per_thread = num_blocks / (double)active_threads;
    std::cout << myrank << ": number of cells: " << num_cells << std::endl;
    std::cout << myrank << ": number of iterations: " << num_iterations << std::endl;
    std::cout << myrank << ": number of active threads: " << active_threads << std::endl;
    std::cout << myrank << ": block size: " << block_size << std::endl;
    std::cout << myrank << ": number of blocks: " << num_blocks << std::endl;
    std::cout << myrank << ": blocks per thread: " << blocks_per_thread << std::endl;
    std::cout << myrank << ": total time in solver: " << p->accumulated << " seconds" << std::endl;
}


/**
 * The Main function
 */
int main (int argc, char ** argv) {
    parse_arguments(argc, argv);

    /* Initialize MPI */

    int required, provided;
    required = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf ("Your MPI installation doesn't allow multiple threads. Exiting.\n");
            exit(0);
    }
    /* Find out my identity in the default communicator */

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    apex::init("MPI TEST", myrank, num_ranks);
    apex_example_set_rank_info(myrank, num_ranks);

#ifdef APEX_HAVE_ACTIVEHARMONY
    //long * inputs[1] = {0L};
    long mins[3] = {1};    // all minimums are 1
    long maxs[3] = {0};    // we'll set these later
    long steps[3] = {1};   // all step sizes are 1
    //inputs[0] = &active_threads;
    mins[0] = apex::apex_options::throttling_min_threads(); // allow user configuration
    maxs[0] = apex::apex_options::throttling_max_threads(); // allow user configuration
    std::cout <<"Tuning Parameters:" << std::endl;
    std::cout <<"\tmins[0]: " << mins[0] << ", maxs[0]: " << maxs[0] << ", steps[0]: " << steps[0] << std::endl;
    my_custom_event = apex::register_custom_event("Perform balance");
    //apex::setup_throughput_tuning((apex_function_address)solve_iteration,
    /*
    apex::setup_throughput_tuning((apex_function_address)barrier_wrapper,
                    APEX_MINIMIZE_ACCUMULATED, my_custom_event, num_inputs,
                    inputs, mins, maxs, steps);
	apex_register_periodic_policy(1000000, apex_periodic_policy_func);
    */
	apex::register_policy(my_custom_event, apex_example_policy_func);
	apex_example_set_function_address((apex_function_address)(solve_iteration));
    //apex_global_setup(APEX_FUNCTION_ADDRESS, (void*)&solve_iteration);
    long original_active_threads = active_threads;
#else
    long original_active_threads = active_threads;
    std::cerr << "Active Harmony not enabled" << std::endl;
#endif
    std::cout <<"Running 1D stencil test..." << std::endl;

    std::vector<double> * prev = initialize(false);
    std::vector<double> * next = initialize(true);
    std::vector<double> * tmp = prev;
    double prev_accumulated = 0.0;
    for (int i = 0 ; i < num_iterations ; i++) {
        solve_iteration(prev, next);
        MPI_Barrier(MPI_COMM_WORLD);
        //dump_array(next);
        tmp = prev;
        prev = next;
        next = tmp;
        if (i % update_interval == 0 && i > 0) {
            apex_profile * p = apex::get_profile((apex_function_address)solve_iteration);
            if (p != nullptr) {
                double next_accumulated = p->accumulated - prev_accumulated;
                prev_accumulated = p->accumulated;
                std::cout << myrank << " Iteration: " << i << " accumulated: " << next_accumulated << std::endl;
            }
            apex::custom_event(my_custom_event, NULL);
            active_threads = apex_example_get_active_threads();
        }
    }
    //dump_array(tmp);
    report_stats();
    delete(prev);
    delete(next);
    std::cout << "done." << std::endl;
#ifdef APEX_HAVE_TAU
    active_threads = original_active_threads;
#endif
	if (active_threads <= original_active_threads) {
    	std::cout << "Test passed." << std::endl;
	}
  /* Shut down MPI */

    //apex_global_teardown(); // do this before MPI_Finalize
    MPI_Barrier(MPI_COMM_WORLD);
    apex::finalize();
    MPI_Finalize();
}



