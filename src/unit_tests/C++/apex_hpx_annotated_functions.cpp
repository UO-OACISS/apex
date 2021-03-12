#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <atomic>
#include <time.h>
#include <stdint.h>
#include "apex_api.hpp"
#if defined(APEX_HAVE_MPI)
#include <mpi.h>
#endif

uint32_t test_numthreads = 0;
int threads_per_core = 8;
__thread uint64_t guid = 0;
const int num_iterations = 10;
int comm_rank = 0;
int comm_size = 1;

#ifdef DEBUG__
#define __DEBUG_PRINT__ 1
#endif

#ifndef __APPLE__
pthread_barrier_t barrier;
#endif

int nsleep(long miliseconds, int tid)
{
   APEX_UNUSED(tid);
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
#endif
   return nanosleep(&req , &rem);
}

static void init_guid(int tid) {
    guid = ((UINT64_MAX/test_numthreads) * tid);
}

void innerLoop(int *tid) {
    std::shared_ptr<apex::task_wrapper> tt_ptr = apex::new_task(__func__);

    const char * new_label = ((comm_rank % 2 == 0) ? "foo" : "bar");
    apex::update_task(tt_ptr, new_label);
    apex::start(tt_ptr);

    /* do some computation */
	int ret = nsleep(10, *tid); // after - t: 10, af: 0
    if (ret != 0) {
        perror("Error occurred while sleeping");
    }

    /* stop the timer */
    apex::stop(tt_ptr);
}

void* someThread(void* tmp)
{
    /* Register this thread with APEX */
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker-thread %d", *tid);
#ifndef __APPLE__
#ifndef APEX_HAVE_OTF2
    pthread_barrier_wait(&barrier);
#endif
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
    #if defined(APEX_HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    std::cout << "APP: rank " << comm_rank << " of " << comm_size << std::endl;
    apex::init("apex::start unit test", comm_rank, comm_size);
    #else
    apex::init("apex::start unit test", 0, 1);
    #endif
	/* important, to make sure we get correct profiles at the end */
    apex::apex_options::use_screen_output(true);
    /* start a timer */
    apex::profiler* p = apex::start("main");
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
    for (int i = 0 ; i < num_iterations ; i++) {
        innerLoop(&(tids[i]));
    }
    /* wait for the threads to finish */
    for (uint32_t i = 0 ; i < test_numthreads ; i++) {
        pthread_join(thread[i], NULL);
    }
    free(tids);
    free(thread);
    /* stop our main timer */
    apex::stop(p);
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
    #if defined(APEX_HAVE_MPI)
    MPI_Finalize();
    #endif
    return 0;
}

