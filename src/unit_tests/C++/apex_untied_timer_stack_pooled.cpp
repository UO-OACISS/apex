// C++ Program to demonstrate thread pooling
// from https://www.geeksforgeeks.org/thread-pool-in-cpp/

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include "apex_api.hpp"
//using namespace std;

static std::mutex cout_mtx;

// Class that represents a simple thread pool
class ThreadPool {
public:
    // // Constructor to creates a thread pool with given
    // number of threads
    ThreadPool(size_t num_threads
               = std::thread::hardware_concurrency()) :
               working(0)
    {

        // Creating worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    // The reason for putting the below code
                    // here is to unlock the queue before
                    // executing the task so that other
                    // threads can perform enqueue tasks
                    {
                        // Locking the queue so that data
                        // can be shared safely
                        std::unique_lock<std::mutex> lock(
                            queue_mutex_);

                        // Waiting until there is a task to
                        // execute or the pool is stopped
                        cv_.wait(lock, [this] {
                            return !tasks_.empty() || stop_;
                        });

                        // exit the thread in case the pool
                        // is stopped and there are no tasks
                        if (stop_ && tasks_.empty()) {
                            return;
                        }

                        // Get the next task from the queue
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    working++;
                    task();
                    working--;
                }
            });
        }
    }

    // Destructor to stop the thread pool - now in "stop"
    ~ThreadPool() { }

    // Enqueue task for execution by the thread pool
    void enqueue(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
        }
        cv_.notify_one();
    }

    void drain() {
        size_t work_left{0};
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            work_left = tasks_.size();
        }
        while(work_left > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::unique_lock<std::mutex> lock(queue_mutex_);
            work_left = tasks_.size();
        }
        while(working > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    void stop() {
        {
            // Lock the queue to update the stop flag safely
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }

        // Notify all threads
        cv_.notify_all();

        // Joining all worker threads to ensure they have
        // completed their tasks
        for (auto& thread : threads_) {
            thread.join();
        }
    }

private:
    // Vector to store worker threads
    std::vector<std::thread> threads_;

    // Queue of tasks
    std::queue<std::function<void()> > tasks_;

    // Mutex to synchronize access to shared data
    std::mutex queue_mutex_;

    // Condition variable to signal changes in the state of
    // the tasks queue
    std::condition_variable cv_;

    // Flag to indicate whether the thread pool should stop
    // or not
    bool stop_ = false;

    // number of actively working workers
    std::atomic<uint64_t> working;
};

uint64_t getid() {
    static std::atomic<uint64_t> id{0};
    uint64_t tmp = ++id;
    return tmp;
}

ThreadPool& pool() {
    // Create a thread pool with 8 threads
    static ThreadPool _pool(8);
    return _pool;
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
    std::cout << "Stopping " << __func__ << std::endl;
    TIMER_STOP
    APEX_ASSERT(tw->prof != nullptr);
    apex::stop(tw);
}

void loop_it(std::shared_ptr<apex::task_wrapper> tw) {
    TIMER_START_PARENT
    pool().enqueue([=] {
        auto tw2 = apex::new_task("task that is on one thread", getid(), _tw);
        simple_task(tw2);
    });
    auto tw5 = apex::new_task("task that is spread over two threads", getid(), _tw);
    pool().enqueue([=] { transient_task_start(tw5); });
    pool().enqueue([=] {
        auto tw3 = apex::new_task("task that is on one thread with parent", getid(), _tw);
        simple_task_with_parent(tw3);
    });
    auto tw4 = apex::new_task("task that is (possibly) stopped elsewhere", getid(), _tw);
    pool().enqueue([=] { transient_task_start(tw4); });
    pool().enqueue([=] {
        // wait for the task to yield...
        while(tw4->state != apex::task_wrapper::YIELDED) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        // ...and wait for the previous profiler to get ingested
        while(tw4->prof != nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        transient_task_stop(tw4);
    });
    pool().enqueue([=] {
        // wait for the task to yield...
        while(tw5->state != apex::task_wrapper::YIELDED) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        // ...and wait for the previous profiler to get ingested
        while(tw5->prof != nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        transient_task_stop(tw5);
    });
    TIMER_STOP
}

int main()
{
    apex::init("apex::init unit test", 0, 1);
    apex::apex_options::use_screen_output(true);
    TIMER_START

    // Enqueue tasks for execution
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            pool().enqueue([=] {
                std::stringstream ss;
                ss << "Task " << j << " is running on thread "
                    << std::this_thread::get_id();
                std::scoped_lock l(cout_mtx);
                std::cout << ss.rdbuf() << std::endl;
                loop_it(_tw);
            });
        }
        pool().drain();
    }
    pool().stop();
    TIMER_STOP
    apex::finalize();
    apex::cleanup();
    return 0;
}