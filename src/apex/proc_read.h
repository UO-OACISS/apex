#ifndef PROC_READ_H
#define PROC_READ_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <thread>
#if APEX_STATIC_BUILD
#include <pthread.h>
#include <sys/time.h>
#endif

namespace apex {

class CPUStat {
public:
  char name[32];
  long long user;
  long long nice;
  long long system;
  long long idle;
  long long iowait;
  long long irq;
  long long softirq;
  long long steal;
  long long guest;
};

typedef std::vector<CPUStat*> CPUs;

class proc_data_reader {
private:
  std::atomic<bool> done;
#if APEX_STATIC_BUILD
  pthread_t proc_reader_thread;
  pthread_mutex_t _my_mutex; // for initialization, termination
  pthread_cond_t _my_cond; // for timer
#else
  std::thread * proc_reader_thread;
  std::condition_variable cv; // for interrupting reader when done
  std::mutex cv_m;            // mutex for the condition variable
#endif
public:
  static void* read_proc(void * _pdr);
  proc_data_reader(void) : done(false) {
#if APEX_STATIC_BUILD
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
    int ret = pthread_create(&proc_reader_thread, NULL, &proc_data_reader::read_proc, (void*)(this));
    if (ret != 0) {
        errno = ret;
        perror("Error: pthread_create (1) fails\n");
        exit(1);
    }
#else
    proc_reader_thread = new std::thread(proc_data_reader::read_proc, (void*)(this));
#endif
  };

  void stop_reading(void) {
#if APEX_STATIC_BUILD
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    pthread_cond_signal(&_my_cond);
    int ret = pthread_join(proc_reader_thread, NULL);
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
#else
    {
        std::unique_lock<std::mutex> lk(cv_m);
        done = true;
    }
    cv.notify_one(); // interrupt the reader thread if it is sleeping!
    proc_reader_thread->join();
#endif
  }

  ~proc_data_reader(void) {
    stop_reading();
#if APEX_STATIC_BUILD
    pthread_cond_destroy(&_my_cond);
    pthread_mutex_destroy(&_my_mutex);
#endif
  }

  bool wait() {
      if (done) return false;
#if APEX_STATIC_BUILD
        struct timespec ts;
        struct timeval  tp;
        gettimeofday(&tp, NULL);
        ts.tv_sec  = (tp.tv_sec + 1);
        ts.tv_nsec = (1000 * tp.tv_usec);
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
#else
    // sleep until next time
    std::unique_lock<std::mutex> lk(cv_m);
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1500) 
    // for some reason, the Intel compiler didn't implement std::cv_status in a normal way.
    // for intel 15, it is in the tbb::interface5 namespace.
    // enum cv_status { no_timeout, timeout }; 
    auto stat = cv.wait_for(lk, std::chrono::seconds(1));
    // assume the enum starts at 0?
    if (stat == 0) { return false; }; // if we were signalled, exit.
#else
    std::cv_status stat = cv.wait_for(lk, std::chrono::seconds(1));
    if (stat != std::cv_status::timeout) { return false; }; // if we were signalled, exit.
#endif
#endif
    return true;
  }
};

class ProcData {
public:
  CPUs cpus;
  long long ctxt;
  long long btime;
  long processes;
  long procs_running;
  long procs_blocked;
#if defined(APEX_HAVE_CRAY_POWER)
  long power;
  long energy;
  long freshness;
  long generation;
  long power_cap;
  long startup;
  long version;
  std::unordered_map<std::string,std::string> cpuinfo;
  std::unordered_map<std::string,std::string> meminfo;
  std::unordered_map<std::string,double> netdev;
#endif
  //softirq 10953997190 0 1380880059 1495447920 1585783785 15525789 0 12 661586214 0 1519806115
  ~ProcData(void);
  ProcData* diff(const ProcData& rhs);
  void dump(std::ostream& out);
  void dump_mean(std::ostream& out);
  void dump_header(std::ostream& out);
  double get_cpu_user();
  void write_user_ratios(std::ostream& out, double *, int);
  void sample_values();
};

class ProcStatistics {
public:
  int size;
  long long *min, *max, *mean;
  ProcStatistics(int);
  //~ProcStatistics();
  int getSize();
};

void get_popen_data(char *);
ProcData* parse_proc_stat(void);
bool parse_proc_cpuinfo();
bool parse_proc_meminfo();
bool parse_proc_self_status();
bool parse_proc_netdev();
bool parse_sensor_data();

/* Ideally, this will read from RCR. If not available, read it directly. 
   Rather than write the same function seven times for seven different
   filenames, just write once and use a foreach macro to expand it to
   all of the versions we need. */

// energy  freshness  generation  power  power_cap  startup  version
#define FOREACH_APEX_XC30_VALUE(macro) \
    macro(energy,"/sys/cray/pm_counters/energy") \
    macro(freshness,"/sys/cray/pm_counters/freshness") \
    macro(generation,"/sys/cray/pm_counters/generation") \
    macro(power,"/sys/cray/pm_counters/power") \
    macro(power_cap,"/sys/cray/pm_counters/power_cap") \
    macro(startup,"/sys/cray/pm_counters/startup") \
    macro(version,"/sys/cray/pm_counters/version") \

#if defined(APEX_HAVE_CRAY_POWER)
#define apex_macro(name,filename) \
inline int read_##name (void) { \
  int tmpint; \
  std::string tmpstr; \
  std::ifstream infile(filename); \
  if (infile.good()) { \
    while (infile >> tmpint >> tmpstr) { \
      return tmpint; /* return the first value encountered. */ \
    } \
  } \
  return 0; \
}
#else
#define apex_macro(name,filename) \
inline int read_##name (void) { \
  return 0; \
}
#endif

FOREACH_APEX_XC30_VALUE(apex_macro)
#undef apex_macro

#ifdef APEX_HAVE_MSR
void apex_init_msr(void);
void apex_finalize_msr(void);
double msr_current_power_high(void); 
#endif

} 
#endif
