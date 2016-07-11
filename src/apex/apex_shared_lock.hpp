#pragma once

#include <pthread.h>

namespace apex {

class shared_lock {
public:
    shared_lock()
    {
        pthread_rwlock_init(&rwlock, NULL);
    }

    void reader_lock() {
        pthread_rwlock_rdlock(&rwlock);
    }

    void unlock() {
        pthread_rwlock_unlock(&rwlock);
    }

    void writer_lock() {
        pthread_rwlock_wrlock(&rwlock);
    }

private:
    pthread_rwlock_t rwlock;
};
};
