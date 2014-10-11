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

namespace apex {

class semaphore
{
private:
    sem_t the_semaphore;
public:
    semaphore() { sem_init(&the_semaphore, 1, 1); }
    inline void post() { sem_post(&the_semaphore); }
    inline void wait() { sem_wait(&the_semaphore); }
};

}

#else 
// Not posix, so use Boost to build a semaphore.

#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>

namespace apex {

class semaphore
{
private:
    boost::mutex mutex_;
    boost::condition_variable condition_;
    unsigned long count_;

public:
    semaphore()
        : count_()
    {}

    void post()
    {
        boost::mutex::scoped_lock lock(mutex_);
        ++count_;
        condition_.notify_one();
    }

    void wait()
    {
        boost::mutex::scoped_lock lock(mutex_);
        while(!count_)
            condition_.wait(lock);
        --count_;
    }
};

};

#endif 
// end of "No posix"
