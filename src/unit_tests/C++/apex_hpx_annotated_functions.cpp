#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <atomic>
#include <time.h>
#include <stdint.h>
#include "apex_api.hpp"

uint32_t numthreads = 0;
int threads_per_core = 8;
__thread uint64_t guid = 0;
const int num_iterations = 10;

#ifndef __APPLE__
pthread_barrier_t barrier;
#endif

int nsleep(long miliseconds, int tid)
{
   struct timespec req, rem;
   // add some variation
   double randval = 1.0 + (((double)(rand())) / RAND_MAX);
   miliseconds = (int)(miliseconds * randval);

   if(miliseconds > 999)
   {   
        req.tv_sec = (int)(miliseconds / 1000);                            /* Must be Non-Negative */
        req.tv_nsec = (miliseconds - ((long)req.tv_sec * 1000)) * 1000000; /* Must be in range of 0 to 999999999 */
   }   
   else
   {   
        req.tv_sec = 0;                         /* Must be Non-Negative */
        req.tv_nsec = miliseconds * 1000000;    /* Must be in range of 0 to 999999999 */
   }   

/*
    std::stringstream buf;
    buf << "APP: " << tid << ": Computing " << miliseconds << " miliseconds\n"; std::cout << buf.str();
    */
   return nanosleep(&req , &rem);
}

static void init_guid(int tid) {
    guid = ((UINT64_MAX/numthreads) * tid);
}

static uint64_t get_guid() {
    //static std::atomic<uint64_t> guid(0);
    return ++guid;
}

void innerLoop(int *tid) {
    /* Start a timer with a GUID*/
    uint64_t myguid = get_guid();
    void* data_ptr = nullptr;
    //std::stringstream buf;
    //buf << "APP: " << *tid << ": Starting thread " << myguid << "\n"; std::cout << buf.str();
    //apex::profiler* p = apex::start((apex_function_address)&someThread, myguid);
    apex::profiler* p = apex::start(__func__, &data_ptr);

    /* do some computation */
	int ret = nsleep(10, *tid); // after - t: 10, af: 0

	/* Start a timer like an "annotated_function" */
    uint64_t afguid = get_guid();
    void* data_ptr2 = nullptr;
    //buf.str(""); buf.clear();
    //buf << "APP: " << *tid << ": Starting annotated_function " << afguid << "\n"; std::cout << buf.str();
    apex::profiler* af = apex::start("annotated function", &data_ptr2);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 20, af: 10

	/* "yield" the outer task */
    //buf.str(""); buf.clear();
    //buf << "APP: " << *tid << ": Yielding thread " << myguid << "\n"; std::cout << buf.str();
	apex::yield(p);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 20, af: 10 - everyone yielded!

	/* resume our current thread */
    //buf.str(""); buf.clear();
    //buf << "APP: " << *tid << ": Resuming thread " << myguid << "\n"; std::cout << buf.str();
    //p = apex::start((apex_function_address)&someThread, myguid);
    p = apex::start(__func__, &data_ptr);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 30, af: 20

    /* stop the annotated_function */
    //buf.str(""); buf.clear();
    //buf << "APP: " << *tid << ": Stopping annotated_function " << afguid << "\n"; std::cout << buf.str();
    apex::stop(af);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 40, af: 20

    /* stop the timer */
    //buf.str(""); buf.clear();
    //buf << "APP: " << *tid << ": Stopping thread " << myguid << "\n"; std::cout << buf.str();
    apex::stop(p);
}

void* someThread(void* tmp)
{
    /* Register this thread with APEX */
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker thread %d", *tid);
#ifndef __APPLE__
    pthread_barrier_wait(&barrier);
#endif
    apex::register_thread(name);
    init_guid(*tid);

    apex::profiler* p = apex::start(__func__);
    for (int i = 0 ; i < num_iterations ; i++) {
        innerLoop(tid);
    }
    apex::stop(p);

    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
	/* important, to make sure we get correct profiles at the end */
    apex::apex_options::use_screen_output(true); 
    /* start a timer */
    apex::profiler* p = apex::start("main");
    /* Spawn X threads */
    if (argc > 1) {
        numthreads = strtoul(argv[1],NULL,0);
    } else {
        numthreads = apex::hardware_concurrency() * threads_per_core; // many threads per core. Stress it!
    }
#ifndef __APPLE__
    pthread_barrier_init(&barrier, NULL, numthreads);
#endif
    if (apex::apex_options::use_tau() || apex::apex_options::use_otf2()) { 
        numthreads = std::min(numthreads, apex::hardware_concurrency());
    }
    int * tids = (int*)calloc(numthreads, sizeof(int));
    pthread_t * thread = (pthread_t*)calloc(numthreads, sizeof(pthread_t));
    for (uint32_t i = 0 ; i < numthreads ; i++) {
        tids[i] = i;
        pthread_create(&(thread[i]), NULL, someThread, &(tids[i]));
    }
    /* wait for the threads to finish */
    for (uint32_t i = 0 ; i < numthreads ; i++) {
        pthread_join(thread[i], NULL);
    }
    free(tids);
    free(thread);
    /* stop our main timer */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
  	apex_profile * profile1 = apex::get_profile("annotated function");
  	apex_profile * profile2 = apex::get_profile("innerLoop");
  	if (profile1 && profile2) {
    	std::cout << "annotated function reported calls : " << profile1->calls << std::endl;
    	std::cout << "innerLoop          reported calls : " << profile2->calls << std::endl;
    	if (profile1->calls == num_iterations * numthreads &&
    	    profile1->calls == profile1->calls) {
        	std::cout << "Test passed." << std::endl;
    	}
  	}
  	apex::cleanup();
    return 0;
}

