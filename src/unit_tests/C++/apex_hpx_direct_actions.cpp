#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <atomic>
#include <time.h>
#include <stdint.h>
#include "apex_api.hpp"
#include "apex_assert.h"

uint32_t test_numthreads = 0;
int threads_per_core = 8;
const int num_iterations = 10;

#ifdef DEBUG
#define __DEBUG_PRINT__ 1
#endif

#ifndef __APPLE__
pthread_barrier_t barrier;
#endif

std::atomic<uint64_t> guid{32};

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

#ifdef __DEBUG_PRINT__
    std::stringstream buf;
    buf << "APP: " << tid << ": Computing " << miliseconds << " miliseconds\n"; std::cout << buf.str();
#else
    APEX_UNUSED(tid);
#endif
   return nanosleep(&req , &rem);
}

void innerLoop(int *tid) {
    std::shared_ptr<apex::task_wrapper> tt_ptr = apex::new_task(__func__, guid++);
#ifdef __DEBUG_PRINT__
    std::stringstream buf;
    buf << "APP: " << *tid << ": Starting thread " << tt_ptr->guid << "\n"; std::cout << buf.str();
#else
    APEX_UNUSED(tid);
#endif
    apex::start(tt_ptr);

    /* do some computation */
	int ret = nsleep(10, *tid); // after - t: 10, af: 0
#ifdef NDEBUG
    APEX_UNUSED(ret);
#else
    APEX_ASSERT(ret == 0);
#endif

	/* Start a timer like an "direct_action" */
    std::shared_ptr<apex::task_wrapper> af = apex::new_task("direct_action", guid++, tt_ptr);
#ifdef __DEBUG_PRINT__
    buf.str(""); buf.clear();
    buf << "APP: " << *tid << ": Starting direct_action " << af->guid << "\n"; std::cout << buf.str();
#endif
    apex::start(af);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 20, af: 10

	/* "yield" the outer task */
#ifdef __DEBUG_PRINT__
    buf.str(""); buf.clear();
    buf << "APP: " << *tid << ": Yielding thread " << tt_ptr->guid << "\n"; std::cout << buf.str();
#endif
	apex::yield(tt_ptr);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 20, af: 10 - everyone yielded!

	/* resume our current thread */
#ifdef __DEBUG_PRINT__
    buf.str(""); buf.clear();
    buf << "APP: " << *tid << ": Resuming thread " << tt_ptr->guid << "\n"; std::cout << buf.str();
#endif
    apex::start(tt_ptr);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 30, af: 20

    /* stop the direct_action */
#ifdef __DEBUG_PRINT__
    buf.str(""); buf.clear();
    buf << "APP: " << *tid << ": Stopping direct_action " << af->guid << "\n"; std::cout << buf.str();
#endif
    apex::stop(af);

    /* do some computation */
	ret = nsleep(10, *tid); // after - t: 40, af: 20

    /* stop the timer */
#ifdef __DEBUG_PRINT__
    buf.str(""); buf.clear();
    buf << "APP: " << *tid << ": Stopping thread " << tt_ptr->guid << "\n"; std::cout << buf.str();
#endif
    apex::stop(tt_ptr);
}

void* someThread(void* tmp)
{
    /* Register this thread with APEX */
    int* tid = (int*)tmp;
    char name[32];
    snprintf(name, 32, "worker-thread %d", *tid);
#ifndef __APPLE__
#ifndef APEX_HAVE_OTF2
    pthread_barrier_wait(&barrier);
#endif
#endif
    apex::register_thread(name);

    auto task = apex::new_task(__func__, guid++);
    apex::start(task);
    for (int i = 0 ; i < num_iterations ; i++) {
        innerLoop(tid);
    }
    apex::stop(task);

    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
	/* important, to make sure we get correct profiles at the end */
    apex::apex_options::use_screen_output(true);
    /* disable untied timers! not yet supported with direct actions. */
    //apex::apex_options::untied_timers(false);
    /* start a timer */
    auto task = apex::new_task("main", guid);
    apex::start(task);
    /* Spawn X threads */
    if (argc > 1) {
        test_numthreads = strtoul(argv[1],NULL,0);
    } else {
        test_numthreads = apex::hardware_concurrency() * threads_per_core; // many threads per core. Stress it!
    }
#ifndef __APPLE__
    pthread_barrier_init(&barrier, NULL, test_numthreads);
#endif
    if (apex::apex_options::use_tau() || apex::apex_options::use_otf2()) {
        test_numthreads = std::min(test_numthreads, apex::hardware_concurrency());
    }
    int * tids = (int*)calloc(test_numthreads, sizeof(int));
    pthread_t * thread = (pthread_t*)calloc(test_numthreads, sizeof(pthread_t));
    for (uint32_t i = 0 ; i < test_numthreads ; i++) {
        tids[i] = i;
        pthread_create(&(thread[i]), NULL, someThread, &(tids[i]));
    }
    /* wait for the threads to finish */
    for (uint32_t i = 0 ; i < test_numthreads ; i++) {
        pthread_join(thread[i], NULL);
    }
    free(tids);
    free(thread);
    /* stop our main timer */
    apex::stop(task);
    /* finalize APEX */
    apex::finalize();
  	apex_profile * profile1 = apex::get_profile("direct_action");
  	apex_profile * profile2 = apex::get_profile("innerLoop");
  	if (profile1 && profile2) {
    	std::cout << "direct_action reported calls : " << profile1->calls << std::endl;
    	std::cout << "innerLoop     reported calls : " << profile2->calls << std::endl;
    	if (profile1->calls == num_iterations * test_numthreads &&
    	    profile1->calls == profile1->calls) {
        	std::cout << "Test passed." << std::endl;
    	}
  	}
  	apex::cleanup();
    return 0;
}

