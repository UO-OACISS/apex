//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <thread>

void foo(int i) {
    struct timespec tim, tim2;
    tim.tv_sec = 0;
    // time slows down...
    tim.tv_nsec = 50000000 + (i * 5000000);
    nanosleep(&tim, &tim2);
}

int main (int argc, char *argv[]) {

    int iters = 50;
    int nthreads = 4;

	if (argc == 1) {
		std::cout << "No iterations specified. Using default of " << iters << "." << std::endl;
	} else if (argc == 2) {
	    iters = atoi(argv[1]);
	} else if (argc == 3) {
	    iters = atoi(argv[1]);
	    nthreads = atoi(argv[2]);
	} else {
		std::cout << "Usage: " << argv[0] << " [num iterations] [num threads]" << std::endl;
		exit(0);
	}

    std::cout << "IN MAIN" << std::endl;

    /* Fork a team of threads giving them their own copies of variables */
    for(int i = 0; i < iters; ++i) {
    	std::cout << "Iteration: " << i << std::endl;;
        std::vector<std::thread> threads;
        for(int t = 0; t < nthreads; ++t) {
            std::thread tmp(foo,i);
            threads.push_back(std::move(tmp));
        }
        for(int t = 0; t < nthreads; ++t) {
            threads[t].join();
        }
        threads.clear();
    }

    std::cerr << "Test passed." << std::endl;
    std::cerr << std::endl;

}

