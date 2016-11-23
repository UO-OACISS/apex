//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>

int main (int argc, char *argv[]) {
	int nthreads;

    int final_nthreads_1 = -1;
    int final_chunk_size_1 = -1;
    int final_schedule_1 = -1;

    int final_nthreads_2 = -1;
    int final_chunk_size_2 = -1;
    int final_schedule_2 = -1;

    int iters = 1000;

	if (argc == 1) {
		std::cout << "No iterations specified. Using default of " << iters << "." << std::endl;
	} else if (argc == 2) {
	    iters = atoi(argv[1]);
	} else {
		std::cout << "Usage: " << argv[0] << " [num iterations]" << std::endl;
		exit(0);
	}

    std::cout << "IN MAIN" << std::endl;

    /* Fork a team of threads giving them their own copies of variables */
    for(int i = 0; i < iters; ++i) {
    	//std::cout << "Iteration: " << i << std::endl;;

#pragma omp parallel private(nthreads)
        {
            nthreads = omp_get_num_threads();
			//#pragma omp single
			//{ std::cout << "numthreads: " << nthreads << std::endl; }
            omp_sched_t sched;
            int chunk_size;
            omp_get_schedule(&sched, &chunk_size);
            struct timespec tim, tim2;
            tim.tv_sec = 0;
            long nt_diff = abs(4 - nthreads);
            long cs_diff = abs(8 - chunk_size);
            long sched_penalty = 0;
            if(sched != omp_sched_static) {
                sched_penalty = 8000000;
            }
            tim.tv_nsec = (nt_diff * 2000000) + (cs_diff * 2000000) + sched_penalty + 1000000;
            nanosleep(&tim, &tim2);
            if(i == 99) {
                #pragma omp master
                {
                    final_nthreads_1 = nthreads;   
                    final_chunk_size_1 = chunk_size;
                    final_schedule_1 = sched;
                }
            }
        }


#pragma omp parallel private(nthreads)
        {
            nthreads = omp_get_num_threads();
            omp_sched_t sched;
            int chunk_size;
            omp_get_schedule(&sched, &chunk_size);
            struct timespec tim, tim2;
            tim.tv_sec = 0;
            long nt_diff = abs(8 - nthreads);
            long cs_diff = abs(128 - chunk_size);
            long sched_penalty = 0;
            if(sched != omp_sched_dynamic) {
                sched_penalty = 8000000;
            }
            tim.tv_nsec = (nt_diff * 2000000) + (cs_diff * 2000000) + sched_penalty + 1000000;
            nanosleep(&tim, &tim2);
            if(i == 99) {
                #pragma omp master
                {
                    final_nthreads_2 = nthreads;   
                    final_chunk_size_2 = chunk_size;
                    final_schedule_2 = sched;
                }
            }
        }
    }

    std::cerr << std::endl;
    std::cerr << "Final omp_num_threads for region 1: " << final_nthreads_1 << " (should be 4)" << std::endl;
    std::cerr << "Final omp_schedule for region 1: " << final_schedule_1 << " (should be " << omp_sched_static << ")" << std::endl;
    std::cerr << "Final omp_chunk_size for region 1: " << final_chunk_size_1 << " (should be 8)" << std::endl;

    std::cerr << "Final omp_num_threads for region 2: " << final_nthreads_2 << " (should be 8)" << std::endl;
    std::cerr << "Final omp_schedule for region 2: " << final_schedule_2 << " (should be " << omp_sched_dynamic << ")" << std::endl;
    std::cerr << "Final omp_chunk_size for region 2: " << final_chunk_size_2 << " (should be 128)" << std::endl;


    //if(final_nthreads_1 == 4 && final_nthreads_2 == 8 && final_schedule_1 == omp_sched_static && final_schedule_2 == omp_sched_dynamic && final_chunk_size_1 == 8 && final_chunk_size_2 == 128) {
    if(final_nthreads_1 != final_nthreads_2 || final_schedule_1 != final_schedule_2 || final_chunk_size_1 != final_chunk_size_2) {
        std::cerr << "Test passed." << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
    std::cerr << std::endl;

}

