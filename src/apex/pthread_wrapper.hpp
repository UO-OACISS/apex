//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <atomic>

#define MILLION 1000000
#define APEX_LXK_KITTEN 1

/*
 * This class exists because std::thread crashes when using std::condition_variable
 * at a timed wait. But only when linked statically.
 */

namespace apex {

class pthread_wrapper {
    private:
        std::atomic<bool> done; // flag for termination
        pthread_t worker_thread; // the wrapped pthread
        pthread_mutex_t _my_mutex; // for initialization, termination
        pthread_cond_t _my_cond; // for timer
        void* (*_func)(void*); // the function for the thread
        void* _context_object;
        unsigned int _timeout_microseconds;
    public:
        pthread_wrapper(void*(*func)(void*), void* context, unsigned int timeout_microseconds) : 
                done(false), 
                _func(func), 
                _context_object(context),
                _timeout_microseconds(timeout_microseconds) {
            pthread_mutexattr_t Attr;
            pthread_mutexattr_init(&Attr);
            pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
            int rc;
            if ((rc = pthread_mutex_init(&_my_mutex, &Attr)) != 0) {
                errno = rc;
                perror("pthread_mutex_init error");
                exit(1);
            }
            if ((rc = pthread_cond_init(&_my_cond, NULL)) != 0) {
                errno = rc;
                perror("pthread_cond_init error");
                exit(1);
            }
            int ret = pthread_create(&worker_thread, NULL, _func, (void*)(this));
            if (ret != 0) {
                errno = ret;
                perror("Error: pthread_create (1) fails\n");
                exit(1);
            }
        };

        void* get_context(void) { return _context_object; };

        void stop_thread(void) {
            //pthread_mutex_lock(&_my_mutex);
            done = true;
            //pthread_mutex_unlock(&_my_mutex);
            pthread_cond_signal(&_my_cond);
            int ret = pthread_join(worker_thread, NULL);
            if (ret != 0) {
                switch (ret) {
                    case ESRCH:
                        // already exited.
                        return;
                    case EINVAL:
                        // Didn't exist?
                        return;
                    case EDEADLK:
                        // trying to join with itself?
                        return;
                    default:
                        errno = ret;
                        perror("Warning: pthread_join failed\n");
                        return;
                }
            }
        }

        ~pthread_wrapper(void) {
            stop_thread();
            pthread_cond_destroy(&_my_cond);
            pthread_mutex_destroy(&_my_mutex);
        }

        bool wait() {
            if (done) return false;
#ifdef APEX_LXK_KITTEN
                int seconds = _timeout_microseconds / MILLION;
                int microseconds = _timeout_microseconds % MILLION;
                struct timespec ts;
                ts.tv_sec  = seconds;
                ts.tv_nsec = 1000 * microseconds;
                int rc = nanosleep(&ts, NULL);
                if (rc != 0) return false;
#else
                struct timespec ts;
                struct timeval  tp;
                gettimeofday(&tp, NULL);
                // add our timeout to "now"
                int seconds = _timeout_microseconds / MILLION;
                int microseconds = _timeout_microseconds % MILLION;
                // check for overflow of the microseconds
                int tmp = tp.tv_usec + microseconds;
                if (tmp > MILLION) {
                    tmp = tmp - MILLION;
                    seconds = seconds + 1;
                }
                // convert to seconds and nanoseconds
                ts.tv_sec  = tp.tv_sec + seconds;
                ts.tv_nsec = 1000 * tmp;
                pthread_mutex_lock(&_my_mutex);
                int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
                if (rc == ETIMEDOUT) {
                    return true;
                } else if (rc == EINVAL) {
                    pthread_mutex_unlock(&_my_mutex);
                    return false;
                } else if (rc == EPERM) {
                    return false;
                }
#endif
            return true;
        }
}; // class

}; // namespace

