#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <atomic>
#include <time.h>
#include "apex_api.hpp"

int nsleep(long miliseconds, int tid)
{
   struct timespec req, rem;

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

    std::stringstream buf;
    buf << "APP: " << tid << ": Computing " << miliseconds << " miliseconds\n"; std::cout << buf.str();
   return nanosleep(&req , &rem);
}

static uint64_t get_guid() {
    static std::atomic<uint64_t> guid(0);
    return ++guid;
}

void* someThread(void* tmp)
{
    /* Register this thread with APEX */
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker thread %d", *tid);
    apex::register_thread(name);

    /* Start a timer with a GUID*/
    uint64_t myguid = get_guid();
    std::stringstream buf;
    buf << "APP: " << *tid << ": Starting thread " << myguid << "\n"; std::cout << buf.str();
    //apex::profiler* p = apex::start((apex_function_address)&someThread, myguid);
    apex::profiler* p = apex::start(__func__, myguid);

    /* do some computation */
	int ret = nsleep(1000, *tid); // after - t: 1000, af: 0

	/* Start a timer like an "annotated_function" */
    uint64_t afguid = get_guid();
    buf = std::stringstream();
    buf << "APP: " << *tid << ": Starting annotated_function " << afguid << "\n"; std::cout << buf.str();
    apex::profiler* af = apex::start("annotated function", afguid);

    /* do some computation */
	ret = nsleep(1000, *tid); // after - t: 2000, af: 1000

	/* "yield" the outer task */
    buf = std::stringstream();
    buf << "APP: " << *tid << ": Yielding thread " << myguid << "\n"; std::cout << buf.str();
	apex::yield(p);

    /* do some computation */
	ret = nsleep(1000, *tid); // after - t: 2000, af: 1000 - everyone yielded!

	/* resume our current thread */
    buf = std::stringstream();
    buf << "APP: " << *tid << ": Resuming thread " << myguid << "\n"; std::cout << buf.str();
    //p = apex::start((apex_function_address)&someThread, myguid);
    p = apex::start(__func__, myguid);

    /* do some computation */
	ret = nsleep(1000, *tid); // after - t: 3000, af: 2000

    /* stop the annotated_function */
    buf = std::stringstream();
    buf << "APP: " << *tid << ": Stopping annotated_function " << afguid << "\n"; std::cout << buf.str();
    apex::stop(af);

    /* do some computation */
	ret = nsleep(1000, *tid); // after - t: 4000, af: 2000

    /* stop the timer */
    buf = std::stringstream();
    buf << "APP: " << *tid << ": Stopping thread " << myguid << "\n"; std::cout << buf.str();
    apex::stop(p);
    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
    apex::apex_options::use_screen_output(true);
    /* start a timer */
    apex::profiler* p = apex::start("main");
    /* Spawn X threads */
    int numthreads = 1;
    int * tids = (int*)calloc(numthreads, sizeof(int));
#if 0
    //int numthreads = apex::hardware_concurrency();
    pthread_t * thread = (pthread_t*)calloc(numthreads, sizeof(pthread_t));
    for (int i = 0 ; i < numthreads ; i++) {
        tids[i] = i;
        pthread_create(&(thread[i]), NULL, someThread, &(tids[i]));
    }
    /* wait for the threads to finish */
    for (int i = 0 ; i < numthreads ; i++) {
        pthread_join(thread[i], NULL);
    }
#else
    tids[0] = 0;
    someThread(&(tids[0]));
#endif
    /* stop our main timer */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
    return 0;
}

