#pragma once

#include <thread>
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "utils.hpp"

namespace apex {
namespace treemerge {

class ThreadPool {
public:
    ThreadPool(uint32_t size = my_hardware_concurrency()) :
        nthreads(size) {};
    void Start(void);
    void QueueJob(const std::function<void()>& job);
    void Stop(void);
    bool busy(void);
    uint32_t getNthreads(void) { return nthreads; };

private:
    void ThreadLoop(void);

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
    uint32_t nthreads;
};

} // namespace treemerge
} // namespace apex