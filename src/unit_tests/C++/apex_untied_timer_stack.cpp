#include "apex_api.hpp"
#include <unistd.h>
#include <thread>

uint64_t getid() {
    static std::atomic<uint64_t> id{0};
    uint64_t tmp = ++id;
    return tmp;
}

#define TIMER_START \
    auto _tw = apex::new_task(__func__, getid()); \
    apex::start(_tw); \
    std::cout << "Started " << __func__ << std::endl;

#define TIMER_START_PARENT \
    auto _tw = apex::new_task(__func__, getid(), tw); \
    apex::start(_tw); \
    std::cout << "Started " << __func__ << std::endl;

#define TIMER_STOP \
    std::cout << "Stopping " << __func__ << std::endl; \
    apex::stop(_tw);

void simple_subroutine_1(void) {
    TIMER_START
    TIMER_STOP
}

void simple_subroutine_2(void) {
    TIMER_START
    TIMER_STOP
}

void simple_task(std::shared_ptr<apex::task_wrapper> tw) {
    apex::start(tw);
    TIMER_START
    simple_subroutine_1();
    apex::yield(tw);
    apex::resume(tw);
    simple_subroutine_2();
    TIMER_STOP
    APEX_ASSERT(tw != nullptr && tw->prof != nullptr);
    apex::stop(tw);
}

void simple_task_with_parent(std::shared_ptr<apex::task_wrapper> tw) {
    apex::start(tw);
    TIMER_START_PARENT
    simple_subroutine_1();
    apex::yield(tw);
    apex::resume(tw);
    simple_subroutine_2();
    TIMER_STOP
    APEX_ASSERT(tw != nullptr && tw->prof != nullptr);
    apex::stop(tw);
}

void transient_task_start(std::shared_ptr<apex::task_wrapper> tw) {
    apex::start(tw);
    TIMER_START
    simple_subroutine_1();
    TIMER_STOP
    apex::yield(tw);
}

void transient_task_stop(std::shared_ptr<apex::task_wrapper> tw) {
    apex::start(tw);
    TIMER_START_PARENT
    simple_subroutine_2();
    TIMER_STOP
    APEX_ASSERT(tw->prof != nullptr);
    apex::stop(tw);
}

void simple_transient_task_stop(std::shared_ptr<apex::task_wrapper> tw) {
    apex::resume(tw);
    TIMER_START_PARENT
    simple_subroutine_2();
    std::cout << "Stopping " << __func__ << std::endl;
    TIMER_STOP
    APEX_ASSERT(tw->prof != nullptr);
    apex::stop(tw);
}

void loop_it(std::shared_ptr<apex::task_wrapper> tw) {
    TIMER_START_PARENT
    {
        auto tw = apex::new_task("task that is on one thread", getid());
        std::thread first = std::thread(simple_task, tw);
        first.join();
    }
    {
        auto tw = apex::new_task("task that is on one thread with parent", getid());
        std::thread first = std::thread(simple_task_with_parent, tw);
        first.join();
    }
    {
        auto tw = apex::new_task("task that is stopped elsewhere", getid());
        std::thread first = std::thread(transient_task_start, tw);
        first.join();
        std::thread second = std::thread(simple_transient_task_stop, tw);
        second.join();
    }
    {
        auto tw = apex::new_task("task that is spread over two threads", getid());
        std::thread first = std::thread(transient_task_start, tw);
        first.join();
        std::thread second = std::thread(transient_task_stop, tw);
        second.join();
    }
    TIMER_STOP
}

void outer_loop_it(std::shared_ptr<apex::task_wrapper> tw) {
    TIMER_START_PARENT
    std::vector<std::thread> threads;
    for (int i = 0 ; i < 5 ; i++) {
        threads.push_back(std::thread(loop_it, _tw));
    }
    for (auto& t : threads) {
        t.join();
    }
    TIMER_STOP
}

int main (int argc, char** argv) {
    APEX_UNUSED(argc);
    APEX_UNUSED(argv);
    apex::init("apex::init unit test", 0, 1);
    apex::apex_options::use_screen_output(true);
    TIMER_START
    std::vector<std::thread> threads;
    for (int i = 0 ; i < 5 ; i++) {
        threads.push_back(std::thread(outer_loop_it, _tw));
    }
    for (auto& t : threads) {
        t.join();
    }
    TIMER_STOP
    apex::finalize();
    apex::cleanup();
    return 0;
}

