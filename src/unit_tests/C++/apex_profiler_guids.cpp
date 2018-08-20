#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <set>
#include <mutex>
#include "apex_api.hpp"

std::set<uint64_t> guids;
std::mutex guid_mutex;
const int num_threads = 1<<6;
const int num_tasks = 1<<8;

void check_guid(uint64_t guid) {
    return;
    std::unique_lock<std::mutex> l(guid_mutex);
#ifdef __VERBOSE_OUTPUT__
    // output like this so that we don't get interleaved output
    std::stringstream ss;
    ss << "Created task with guid: " << guid << "\n";
    std::cout << ss.str(); fflush(stdout);
#endif
    if (guids.find(guid) != guids.end()) {
        std::cerr << "Error! Duplicated GUID!" << std::endl;
        abort();
    }
    guids.insert(guid);
}

int foo(int input) {
    apex::profiler* p = apex::start((apex_function_address)&foo);
    check_guid(p->guid);
    int output = input * input;
    apex::stop(p);
    return output;
}

void* someThread(void* tmp)
{
    int tid = *(int*)tmp;
    char name[32];
    sprintf(name, "worker thread %d", tid);
    /* Register this thread with APEX */
    apex::register_thread(name);
    /* Start a timer */
    apex::profiler* p = apex::start((apex_function_address)&someThread);
    check_guid(p->guid);
    /* ... */
    /* do some computation */
    for (int i = 0 ; i < num_tasks ; i++) {
        foo(i+tid);
    }
    /* ... */
    /* stop the timer */
    apex::stop(p);
    /* tell APEX that this thread is exiting */
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    /* initialize APEX */
    apex::init("apex::start unit test", 0, 1);
    /* start a timer */
    apex::profiler* p = apex::start("main");
    check_guid(p->guid);
    /* Spawn two threads */
    pthread_t thread[num_threads];
    int tids[num_threads];
    for (int i = 0 ; i < num_threads ; i++) {
        tids[i] = i;
        pthread_create(&(thread[i]), NULL, someThread, &(tids[i]));
    }
    for (int i = 0 ; i < num_threads ; i++) {
        /* wait for the thread to finish. Not concurrent, but tests task guid generation. */
        pthread_join(thread[i], NULL);
    }
    /* stop our main timer */
    apex::stop(p);
    /* finalize APEX */
    apex::finalize();
    return 0;
}

