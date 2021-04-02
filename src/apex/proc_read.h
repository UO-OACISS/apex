/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#if APEX_HAVE_PROC

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
#include <string>
#include <memory>
#include "pthread_wrapper.hpp"
#include "apex_options.hpp"

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
    pthread_wrapper * worker_thread;
    static std::atomic<bool> done;
public:
    static void* read_proc(void * _pdr);
    proc_data_reader(void) {
        worker_thread = new pthread_wrapper(&proc_data_reader::read_proc,
        (void*)(this), apex_options::proc_period());
    };

    void stop_reading(void) {
        done = true;
        worker_thread->stop_thread();
    }

    ~proc_data_reader(void) {
        stop_reading();
        delete worker_thread;
    }
    static std::string get_command_line(void);
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
#if defined(APEX_HAVE_POWERCAP_POWER)
  long long package0;
  long long dram;
#endif
#if defined(APEX_HAVE_PAPI)
  std::vector<long long> rapl_metrics;
  std::vector<long long> nvml_metrics;
  std::vector<long long> lms_metrics;
#endif
  //softirq 10953997190 0 1380880059 1495447920 1585783785 ...
  //        15525789 0 12 661586214 0 1519806115
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

#if defined(APEX_HAVE_POWERCAP_POWER)
/*
 * This isn't really the right way to do this. What should be done
 * is:
 * 1) read /sys/class/powercap/intel-rapl/intel-rapl:0/name to get the counter name (once)
 * 2) read /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj to get the value
 * 3) for i in /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:*:*
 *    do 1), 2) above for each
 *
 * This was a quick hack to get basic support for KNL.
 */
inline long long read_package0 (void) {
  long long tmplong;
  FILE *fff;
  fff=fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj","r");
  if (fff==nullptr) {
    //std::cerr << "Error opening package0!" << std::endl;
    tmplong = 0LL;
  } else {
    fscanf(fff,"%lld",&tmplong);
    fclose(fff);
  }
  return tmplong/1000000;
}

inline long long  read_dram (void) {
  //std::cout << "Reading dram" << std::endl;
  long long  tmplong;
  FILE *fff;
  fff=fopen(
    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
    "r");
  if (fff==nullptr) {
    //std::cerr << "Error opening dram!" << std::endl;
    tmplong = 0LL;
  } else {
    fscanf(fff,"%lld",&tmplong);
    fclose(fff);
  }
  return tmplong/1000000;
}

#endif

#if defined(APEX_HAVE_PAPI)
void initialize_papi_events(void);
void read_papi_components(ProcData * data);
#endif

#ifdef APEX_HAVE_MSR
void apex_init_msr(void);
void apex_finalize_msr(void);
double msr_current_power_high(void);
#endif

}

#endif
