#ifndef PROC_READ_H
#define PROC_READ_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>

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

class ProcData {
public:
  CPUs cpus;
  long long ctxt;
  long long btime;
  long processes;
  long procs_running;
  long procs_blocked;
  long power;
  long energy;
  //softirq 10953997190 0 1380880059 1495447920 1585783785 15525789 0 12 661586214 0 1519806115
  ~ProcData();
  ProcData* diff(const ProcData& rhs);
  void dump(std::ostream& out);
  void dump_mean(std::ostream& out);
  void dump_header(std::ostream& out);
  double get_cpu_user();
  void write_user_ratios(std::ostream& out, double *, int);
  void sample_values();
  static void read_proc(void);
  static void stop_reading(void);

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

/* Ideally, this will read from RCR. If not available, read it directly. */
inline int read_power(void) {
#if defined(APEX_HAVE_CRAY_POWER)
  // on the cray XC30, we can read the power in watts
  int tmpint;
  std::string tmpstr;
  std::ifstream infile("/sys/cray/pm_counters/power");
  while (infile >> tmpint >> tmpstr) {
    return tmpint; // return the first value encountered.
  }
#endif
  return 0;
}

/* Ideally, this will read from RCR. If not available, read it directly. */
inline int read_energy(void) {
#if defined(APEX_HAVE_CRAY_POWER)
  // on the cray XC30, we can read the energy in Joules
  int tmpint;
  std::string tmpstr;
  std::ifstream infile2("/sys/cray/pm_counters/energy");
  while (infile2 >> tmpint >> tmpstr) {
    return tmpint; // return the first value encountered.
  }
#endif
  return 0;
}

} 
#endif
