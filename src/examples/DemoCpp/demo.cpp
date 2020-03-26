#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace std;

/* First method for timers: using a scoped timer object.
 * This is the simplest method for timing C++ */

void scopedTimerExample1(void) {
    /* This function is timed with a scoped timer using the function name */
    apex::scoped_timer(__func__);
    usleep(2000);
    return;
}

void scopedTimerExample2(void) {
    /* This function is timed with a scoped timer using the function address */
    apex::scoped_timer((apex_function_address)&scopedTimerExample2);
    usleep(2000);
    return;
}

/* Second method for timers: using a task wrapper.
 * The benefit of this case is that the timer can be yielded and resumed */

void taskWrapperExample1(void) {
    /* This function is timed with a task_wrapper that is explicitly started/stopped */
    std::shared_ptr<apex::task_wrapper> wrapper = apex::new_task(__func__);
    apex::start(wrapper);
    usleep(2000);
    apex::yield(wrapper);
    usleep(2000);
    apex::start(wrapper);
    usleep(2000);
    apex::stop(wrapper);
    return;
}

void taskWrapperExample2(void) {
    /* This function is timed with a task_wrapper that is explicitly started/stopped */
    std::shared_ptr<apex::task_wrapper> wrapper =
        apex::new_task((apex_function_address)&taskWrapperExample2);
    apex::start(wrapper);
    usleep(2000);
    apex::yield(wrapper);
    usleep(2000);
    apex::start(wrapper);
    usleep(2000);
    apex::stop(wrapper);
    return;
}

/* Third example, using simple profiler objects */

void profilerExample1(void) {
    /* This function is timed with a profiler object */
    auto * profiler = apex::start(__func__);
    usleep(2000);
    apex::stop(profiler);
}

void profilerExample2(void) {
    /* This function is timed with a profiler object */
    auto * profiler = apex::start((apex_function_address)&profilerExample2);
    usleep(2000);
    apex::stop(profiler);
}

void* someThread(void* tmp)
{
    int* tid = (int*)tmp;
    char name[32];
    sprintf(name, "worker-thread#%d", *tid);
    /* Tell APEX that there is a new thread */
    apex::register_thread(name);
    /* Time this thread */
    apex::profiler* p = apex::start((apex_function_address)someThread);
    /* Sample a counter */
    apex::sample_value("test_counter_1", 2.0);
    char counter[64];
    /* Sample another counter */
    sprintf(counter, "test_counter_%s", name);
    apex::sample_value(counter, 2.0);
    /* Stop timing the thread */
    apex::stop(p);
    apex::exit_thread();
    return NULL;
}

int main (int argc, char** argv) {
    APEX_UNUSED(argc);
    /* Initialize APEX */
    apex::init(argv[0], 0, 1);
    /* Get some version and verbose option information */
    cout << "APEX Version : " << apex::version() << endl;
    apex::apex_options::print_options();
    apex::apex_options::use_screen_output(true);
    /* Start a timer for the main function, using its address (requires binutils) */
    apex::profiler* p = apex::start((apex_function_address)(main));
    /* Launch two threads */
    pthread_t thread[2];
    int tid = 0;
    pthread_create(&(thread[0]), NULL, someThread, &tid);
    int tid2 = 1;
    pthread_create(&(thread[1]), NULL, someThread, &tid2);
    /* Join the 2 threads */
    pthread_join(thread[0], NULL);
    pthread_join(thread[1], NULL);

    /* test some other timers */
    scopedTimerExample1();
    scopedTimerExample2();
    taskWrapperExample1();
    taskWrapperExample2();
    profilerExample1();
    profilerExample2();

    /* Stop the main timer and exit */
    apex::stop(p);
    apex::finalize();
    return 0;
}

