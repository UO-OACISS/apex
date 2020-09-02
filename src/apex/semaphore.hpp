/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

/* Apparently, Boost does not have semaphores. So, we implement one.
 * Example from:
 * http://stackoverflow.com/questions/4792449/c0x-has-no-semaphores-how-to-synchronize-threads
 */

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || \
    (defined(__APPLE__) && defined(__MACH__)))
    /* UNIX-style OS. ------------------------------------------- */
#include <unistd.h>
#include <iostream>
#endif

#if defined(_POSIX_VERSION) && !defined(_MSC_VER) && !defined(__APPLE__)
// Posix!

#include <semaphore.h>
#include <atomic>

namespace apex {

class semaphore
{
private:
#if !defined(__APPLE__) // handle the new way on systems like Apple
    sem_t the_semaphore;
#endif
    sem_t * the_semaphore_p;
    std::atomic<bool> work_waiting;
    std::atomic<uint64_t> posts;
    std::atomic<uint64_t> true_posts;
    std::atomic<uint64_t> waits;
public:
    semaphore() : work_waiting(0), posts(0), true_posts(0), waits(0) {
#if defined(__APPLE__) // handle the new way on systems like Apple
        the_semaphore_p = sem_open("work waiting", O_CREAT, S_IRWXU, 1);
#else
        the_semaphore_p = &the_semaphore; sem_init(the_semaphore_p, 1, 1);
#endif
    }
    void dump_stats() {
#ifdef APEX_DEBUG
        std::cout << "Semaphore stats: " << posts << " posts, "
        << true_posts << " true posts, " << waits << " waits."
        << std::endl; fflush(stdout);
#endif
    }
    /*
     * This function is somewhat optimized. Because we were spending a lot of time
     * waiting for the post (it is a synchronization point across all threads), don't
     * post if there is already work on the queue.
     */
    inline void post() {
        //posts++;
        if (work_waiting) return ;
        work_waiting = true;
        //true_posts++;
        sem_post(the_semaphore_p);
    }
    /*
     * When the wait is over, clear the "work_waiting" flag, even though we haven't
     * cleared the waiting profilers.
     */
    inline void wait() { sem_wait(the_semaphore_p);
        //waits++;
        work_waiting=false;
    }
};

}

#else
// Not posix, so use std to build a semaphore.

#include <condition_variable>
#include <mutex>

namespace apex {

class semaphore
{
private:
    std::mutex mutex_;
    std::condition_variable_any condition_;
    //unsigned long count_;
    bool work_waiting;

public:
    semaphore()
        //: count_()
        : work_waiting(false)
    {}

    void dump_stats() { }

    void post()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        //++count_;
        work_waiting = true;
        condition_.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        //while(!count_)
        while(!work_waiting)
            condition_.wait(lock);
        work_waiting = false;
        //--count_;
    }
};

};

#endif
// end of "No posix"
