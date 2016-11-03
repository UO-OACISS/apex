//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

/* Apparently, Boost does not have semaphores. So, we implement one.
 * Example from:
 * http://stackoverflow.com/questions/4792449/c0x-has-no-semaphores-how-to-synchronize-threads
 */

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
    /* UNIX-style OS. ------------------------------------------- */
#include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
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
    //int work_waiting;
    std::atomic<bool> work_waiting;
public:
#if defined(__APPLE__) // handle the new way on systems like Apple
    semaphore() : work_waiting(0) { the_semaphore_p = sem_open("work waiting", O_CREAT, S_IRWXU, 1); }
#else
    semaphore() : work_waiting(0) { the_semaphore_p = &the_semaphore; sem_init(the_semaphore_p, 1, 1); }
#endif
    /*
     * This function is somewhat optimized. Because we were spending a lot of time
     * waiting for the post (it is a synchronization point across all threads), don't
     * post if there is already work on the queue.
     */
    inline void post() { if (work_waiting) return ;
        //__sync_fetch_and_add(&work_waiting, 1) ;
        work_waiting = true;
        sem_post(the_semaphore_p);
        }
    /*
     * When the wait is over, clear the "work_waiting" flag, even though we haven't
     * cleared the waiting profilers.
     */
    inline void wait() { sem_wait(the_semaphore_p);
        //__sync_fetch_and_sub(&work_waiting, work_waiting); }
        work_waiting=false; }
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
    unsigned long count_;

public:
    semaphore()
        : count_()
    {}

    void post()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        condition_.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while(!count_)
            condition_.wait(lock);
        --count_;
    }
};

};

#endif
// end of "No posix"
