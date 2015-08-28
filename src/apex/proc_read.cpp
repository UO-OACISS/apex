#if defined(APEX_HAVE_PROC)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "proc_read.h"
#include "apex_api.hpp"
#include "apex.hpp"
#include <boost/atomic.hpp>
#include <sstream>
#include <string>
#include <boost/regex.hpp>
#include <set>
#include "utils.hpp"
#include <condition_variable>
#include <chrono>
#include <atomic>

#define COMMAND_LEN 20
#define DATA_SIZE 512

#define APEX_GET_ALL_CPUS 0

#ifdef APEX_HAVE_TAU
#define PROFILING_ON
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>
#endif

#ifdef APEX_HAVE_LM_SENSORS
#include "sensor_data.hpp"
#endif

std::condition_variable cv; // for interrupting reader when done
std::mutex cv_m;            // mutex for the condition variable

using namespace std;

namespace apex {

/* Flag indicating that we are done, so the
 * reader knows when to exit */
static boost::atomic<bool> proc_done (false);

void get_popen_data(char *cmnd) {
    FILE *pf;
    string command;
    char data[DATA_SIZE];
 
    // Execute a process listing
    command = "cat /proc/"; 
    command = command + cmnd; 
 
    // Setup our pipe for reading and execute our command.
    pf = popen(command.c_str(),"r"); 
 
    if(!pf){
      cerr << "Could not open pipe for output." << endl;
      return;
    }
 
    // Grab data from process execution
    while ( fgets( data, DATA_SIZE, pf)) {
      cout << "-> " << data << endl; 
      fflush(pf);
    }
 
    if (pclose(pf) != 0) {
        cerr << "Error: Failed to close command stream" << endl;
    }
 
    return;
}

ProcData* parse_proc_stat(void) {
  if (!apex_options::use_proc_stat()) return NULL;

  /*  Reading proc/stat as a file  */
  FILE * pFile;
  char line[128];
  char dummy[32];
  pFile = fopen ("/proc/stat","r");
  ProcData* procData = new ProcData();
  if (pFile == NULL) perror ("Error opening file");
  else {
    CPUStat* cpu_stat;
    while ( fgets( line, 128, pFile)) {
      if ( strncmp (line, "cpu", 3) == 0 ) { 
        cpu_stat = new CPUStat();
        /*  Note, this will only work on linux 2.6.24 through 3.5  */
        sscanf(line, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld\n", 
            cpu_stat->name, &cpu_stat->user, &cpu_stat->nice, 
            &cpu_stat->system, &cpu_stat->idle, 
            &cpu_stat->iowait, &cpu_stat->irq, &cpu_stat->softirq, 
            &cpu_stat->steal, &cpu_stat->guest);
        procData->cpus.push_back(cpu_stat);
      }
      else if ( strncmp (line, "ctxt", 4) == 0 ) { 
        sscanf(line, "%s %lld\n", dummy, &procData->ctxt);
      } else if ( strncmp (line, "btime", 5) == 0 ) { 
        sscanf(line, "%s %lld\n", dummy, &procData->btime);
      } else if ( strncmp (line, "processes", 9) == 0 ) { 
        sscanf(line, "%s %ld\n", dummy, &procData->processes);
      } else if ( strncmp (line, "procs_running", 13) == 0 ) { 
        sscanf(line, "%s %ld\n", dummy, &procData->procs_running);
      } else if ( strncmp (line, "procs_blocked", 13) == 0 ) { 
        sscanf(line, "%s %ld\n", dummy, &procData->procs_blocked);
      //} else if ( strncmp (line, "softirq", 5) == 0 ) { 
        // softirq 10953997190 0 1380880059 1495447920 1585783785 15525789 0 12 661586214 0 1519806115
        //sscanf(line, "%s %d\n", dummy, &procData->btime);
      }
#if !(APEX_GET_ALL_CPUS)
      // don't waste time parsing anything but the mean
      break;
#endif
    }
  }
  fclose (pFile);
#if defined(APEX_HAVE_CRAY_POWER)
  procData->power = read_power();
  procData->power_cap = read_power_cap();
  procData->energy = read_energy();
  procData->freshness = read_freshness();
  procData->generation = read_generation();
#endif
  return procData;
}

ProcData::~ProcData() {
  while (!cpus.empty()) {
    delete cpus.back();
    cpus.pop_back();
  }
}

ProcData* ProcData::diff(ProcData const& rhs) {
  ProcData* d = new ProcData();
  unsigned int i;
  CPUStat* cpu_stat;
  for (i = 0 ; i < cpus.size() ; i++) {
    cpu_stat = new CPUStat();
    strcpy(cpu_stat->name, cpus[i]->name);
    cpu_stat->user = cpus[i]->user - rhs.cpus[i]->user; 
    cpu_stat->nice = cpus[i]->nice - rhs.cpus[i]->nice;
    cpu_stat->system = cpus[i]->system - rhs.cpus[i]->system;
    cpu_stat->idle = cpus[i]->idle - rhs.cpus[i]->idle;
    cpu_stat->iowait = cpus[i]->iowait - rhs.cpus[i]->iowait;
    cpu_stat->irq = cpus[i]->irq - rhs.cpus[i]->irq;
    cpu_stat->softirq = cpus[i]->softirq - rhs.cpus[i]->softirq;
    cpu_stat->steal = cpus[i]->steal - rhs.cpus[i]->steal;
    cpu_stat->guest = cpus[i]->guest - rhs.cpus[i]->guest;
    d->cpus.push_back(cpu_stat);
  }
  d->ctxt = ctxt - rhs.ctxt;
  d->processes = processes - rhs.processes;
  d->procs_running = procs_running - rhs.procs_running;
  d->procs_blocked = procs_blocked - rhs.procs_blocked;
#if defined(APEX_HAVE_CRAY_POWER)
  d->power = power;
  d->power_cap = power_cap;
  d->energy = energy - rhs.energy;
  d->freshness = freshness;
  d->generation = generation;
#endif
  return d;
}

void ProcData::dump(ostream &out) {
  out << "name\tuser\tnice\tsys\tidle\tiowait\tirq\tsoftirq\tsteal\tguest" << endl;
  CPUs::iterator iter;
  long long total = 0L;
  double idle_ratio = 0.0;
  double user_ratio = 0.0;
  double system_ratio = 0.0;
  for (iter = cpus.begin(); iter != cpus.end(); ++iter) {
    CPUStat* cpu_stat=*iter;
    out << cpu_stat->name << "\t" 
        << cpu_stat->user << "\t" 
        << cpu_stat->nice << "\t" 
        << cpu_stat->system << "\t" 
        << cpu_stat->idle << "\t" 
        << cpu_stat->iowait << "\t" 
        << cpu_stat->irq << "\t" 
        << cpu_stat->softirq << "\t" 
        << cpu_stat->steal << "\t" 
        << cpu_stat->guest << endl;
	if (strcmp(cpu_stat->name, "cpu") == 0) {
	  total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq + cpu_stat->steal + cpu_stat->guest;
	  user_ratio = (double)cpu_stat->user / (double)total;
	  system_ratio = (double)cpu_stat->system / (double)total;
	  idle_ratio = (double)cpu_stat->idle / (double)total;
	}
  }
  out << "ctxt " << ctxt << endl;
  out << "processes " << processes << endl;
  out << "procs_running " << procs_running << endl;
  out << "procs_blocked " << procs_blocked << endl;
  out << "user ratio " << user_ratio << endl;
  out << "system ratio " << system_ratio << endl;
  out << "idle ratio " << idle_ratio << endl;
  //out << "softirq %d\n", btime);
}

void ProcData::dump_mean(ostream &out) {
  CPUs::iterator iter;
  iter = cpus.begin();
    CPUStat* cpu_stat=*iter;
    out << cpu_stat->name << "\t" 
        << cpu_stat->user << "\t" 
        << cpu_stat->nice << "\t" 
        << cpu_stat->system << "\t" 
        << cpu_stat->idle << "\t" 
        << cpu_stat->iowait << "\t" 
        << cpu_stat->irq << "\t" 
        << cpu_stat->softirq << "\t" 
        << cpu_stat->steal << "\t" 
        << cpu_stat->guest << endl;
}

void ProcData::dump_header(ostream &out) {
  out << "name\tuser\tnice\tsys\tidle\tiowait\tirq\tsoftirq\tsteal\tguest" << endl;
}

double ProcData::get_cpu_user() {

  CPUs::iterator iter;
  long long total = 0L;
  double user_ratio = 0.0;
  for (iter = cpus.begin(); iter != cpus.end(); ++iter) {
    CPUStat* cpu_stat=*iter;
	  if (strcmp(cpu_stat->name, "cpu") == 0) {
	    total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq + cpu_stat->steal + cpu_stat->guest;
	    user_ratio = (double)cpu_stat->user / (double)total;
      break;
	  }
  }

  return user_ratio;

}

void ProcData::write_user_ratios(ostream &out, double *ratios, int num) {

  for (int i=0; i<num; i++) {
    out << ratios[i] << " ";
  }
  out << endl;

}

ProcStatistics::ProcStatistics(int size) {

  this->min = (long long*) malloc (size*sizeof(long long));
  this->max = (long long*) malloc (size*sizeof(long long));
  this->mean = (long long*) malloc (size*sizeof(long long));
  this->size = size;

}

//ProcStatistics::~ProcStatistics() {
//  free (this->min);
//  free (this->max);
//  free (this->mean);
//}

int ProcStatistics::getSize() {
  return this->size;
}

void ProcData::stop_reading(void) {
  proc_done = true;
  cv.notify_all(); // interrupt the reader thread if it is sleeping!
}

void ProcData::sample_values(void) {
  long long total;
  double idle_ratio;
  double user_ratio;
  double system_ratio;
  CPUs::iterator iter = cpus.begin();
  CPUStat* cpu_stat=*iter;
  // convert all measurements from "Jiffies" to seconds
  /*
  sample_value("CPU User", cpu_stat->user);
  sample_value("CPU Nice", cpu_stat->nice);
  sample_value("CPU System", cpu_stat->system);
  sample_value("CPU Idle", cpu_stat->idle);
  sample_value("CPU I/O Wait", cpu_stat->iowait);
  sample_value("CPU IRQ", cpu_stat->irq);
  sample_value("CPU soft IRQ", cpu_stat->softirq);
  sample_value("CPU Steal", cpu_stat->steal);
  sample_value("CPU Guest", cpu_stat->guest);
  */
  total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq + cpu_stat->steal + cpu_stat->guest;
  total = total / 100.0; // so we have a percentage in the final values
  sample_value("CPU User %", (double)cpu_stat->user / (double)total);
  sample_value("CPU Nice %", (double)cpu_stat->nice / (double)total);
  sample_value("CPU System %", (double)cpu_stat->system / (double)total);
  sample_value("CPU Idle %", (double)cpu_stat->idle / (double)total);
  sample_value("CPU I/O Wait %", (double)cpu_stat->iowait / (double)total);
  sample_value("CPU IRQ %", (double)cpu_stat->irq / (double)total);
  sample_value("CPU soft IRQ %", (double)cpu_stat->softirq / (double)total);
  sample_value("CPU Steal %", (double)cpu_stat->steal / (double)total);
  sample_value("CPU Guest %", (double)cpu_stat->guest / (double)total);
#if defined(APEX_HAVE_CRAY_POWER)
  sample_value("Power", power);
  sample_value("Power Cap", power_cap);
  sample_value("Energy", energy);
  sample_value("Freshness", freshness);
  sample_value("Generation", generation);
#endif
  /* This code below is for detailed measurement from all CPUS. */
#if APEX_GET_ALL_CPUS
  ++iter;
  while (iter != cpus.end()) {
    CPUStat* cpu_stat=*iter;
    sample_value(string(cpu_stat->name) + " User", cpu_stat->user);
    sample_value(string(cpu_stat->name) + " Nice", cpu_stat->nice);
    sample_value(string(cpu_stat->name) + " System", cpu_stat->system);
    sample_value(string(cpu_stat->name) + " Idle", cpu_stat->idle);
    sample_value(string(cpu_stat->name) + " I/O Wait", cpu_stat->iowait);
    sample_value(string(cpu_stat->name) + " IRQ", cpu_stat->irq);
    sample_value(string(cpu_stat->name) + " soft IRQ", cpu_stat->softirq);
    sample_value(string(cpu_stat->name) + " Steal", cpu_stat->steal);
    sample_value(string(cpu_stat->name) + " Guest", cpu_stat->guest);
    total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle;
    user_ratio = (double)cpu_stat->user / (double)total;
    system_ratio = (double)cpu_stat->system / (double)total;
    idle_ratio = (double)cpu_stat->idle / (double)total;
    sample_value(string(cpu_stat->name) + " User Ratio", user_ratio);
    sample_value(string(cpu_stat->name) + " System Ratio", system_ratio);
    sample_value(string(cpu_stat->name) + " Idle Ratio", idle_ratio);
    ++iter;
  }
#endif
}

bool parse_proc_cpuinfo() {
  if (!apex_options::use_proc_cpuinfo()) return false;

  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[4096] = {0};
    int cpuid = 0;
    while ( fgets( line, 4096, f)) {
        string tmp(line);
        const boost::regex separator(":");
        boost::sregex_token_iterator token(tmp.begin(), tmp.end(), separator, -1);
        boost::sregex_token_iterator end;
        string name = *token++;
        if (token != end) {
          string value = *token;
          name = trim(name);
          char* pEnd;
          double d1 = strtod (value.c_str(), &pEnd);
	      if (strcmp(name.c_str(), "processor") == 0) { cpuid = (int)d1; }
          stringstream cname;
          cname << "cpuinfo." << cpuid << ":" << name;
          if (pEnd) { sample_value(cname.str(), d1); }
        }
    }
    fclose(f);
  } else {
    return false;
  }
  return true;
}

bool parse_proc_meminfo() {
  if (!apex_options::use_proc_meminfo()) return false;
  FILE *f = fopen("/proc/meminfo", "r");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        string tmp(line);
        const boost::regex separator(":");
        boost::sregex_token_iterator token(tmp.begin(), tmp.end(), separator, -1);
        boost::sregex_token_iterator end;
        string name = *token++;
        if (token != end) {
            string value = *token;
            char* pEnd;
            double d1 = strtod (value.c_str(), &pEnd);
            string mname("meminfo:" + name);
            if (pEnd) { sample_value(mname, d1); }
        }
    }
    fclose(f);
  } else {
    return false;
  }
  return true;
}

bool parse_proc_self_status() {
  if (!apex_options::use_proc_self_status()) return false;
  FILE *f = fopen("/proc/self/status", "r");
  const std::string prefix("Vm");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        string tmp(line);
        if (!tmp.compare(0,prefix.size(),prefix)) {
            const boost::regex separator(":");
            boost::sregex_token_iterator token(tmp.begin(), tmp.end(), separator, -1);
            boost::sregex_token_iterator end;
            string name = *token++;
            if (token != end) {
                string value = *token;
                char* pEnd;
                double d1 = strtod (value.c_str(), &pEnd);
                string mname("self_status:" + name);
                if (pEnd) { sample_value(mname, d1); }
            }
        }
    }
    fclose(f);
  } else {
    return false;
  }
  return true;
}

bool parse_proc_netdev() {
  if (!apex_options::use_proc_net_dev()) return false;
  FILE *f = fopen("/proc/net/dev", "r");
  if (f) {
    char line[4096] = {0};
    char * rc = fgets(line, 4096, f); // skip this line
    if (rc == NULL) {
        fclose(f);
        return false;
    }
    rc = fgets(line, 4096, f); // skip this line
    if (rc == NULL) {
        fclose(f);
        return false;
    }
    while (fgets(line, 4096, f)) {
        string outer_tmp(line);
        outer_tmp = trim(outer_tmp);
        const boost::regex separator("[|:\\s]+");
        boost::sregex_token_iterator token(outer_tmp.begin(), outer_tmp.end(), separator, -1);
        boost::sregex_token_iterator end;
        string devname = *token++; // device name
        string tmp = *token++;
        char* pEnd;
        double d1 = strtod (tmp.c_str(), &pEnd);
        string cname = devname + ".receive.bytes";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.packets";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.errs";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.drop";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.fifo";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.frame";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.compressed";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".receive.multicast";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.bytes";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.packets";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.errs";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.drop";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.fifo";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.colls";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.carrier";
        sample_value(cname, d1);

        tmp = *token++;
        d1 = strtod (tmp.c_str(), &pEnd);
        cname = devname + ".transmit.compressed";
        sample_value(cname, d1);
    }
    fclose(f);
  } else {
    return false;
  }
  return true;
}

// there will be N devices, with M sensors per device.
bool parse_sensor_data() {
#if 0
  string prefix = "/Users/khuck/src/xpress-apex/proc/power/";
  // Find out how many devices have sensors
  std::set<string> devices;
  devices.append(string("coretemp.0"));
  devices.append(string("coretemp.1"));
  devices.append(string("i5k_amb.0"));
  for (std::unordered_set<string>::const_iterator it = devices.begin(); it != devices.end(); it++) {
    // for each device, find out how many sensors there are.
  }
#endif
  return true;
}

/* This is the main function for the reader thread. */
void ProcData::read_proc(void) {
  static bool _initialized = false;
  if (!_initialized) {
      initialize_worker_thread_for_TAU();
      _initialized = true;
  }
#ifdef APEX_HAVE_TAU
  if (apex_options::use_tau()) {
    TAU_START("ProcData::read_proc");
  }
#endif
#ifdef APEX_HAVE_LM_SENSORS
  sensor_data * mysensors = new sensor_data();
#endif
  ProcData *oldData = parse_proc_stat();
  // disabled for now - not sure that it is useful
  parse_proc_cpuinfo(); // do this once, it won't change.
  parse_proc_meminfo(); // some things change, others don't...
  parse_proc_self_status(); // some things change, others don't...
  parse_proc_netdev();
#ifdef APEX_HAVE_LM_SENSORS
  mysensors->read_sensors();
#endif
  ProcData *newData = NULL;
  ProcData *periodData = NULL;
  
  while(!proc_done) {
    // sleep until next time
    std::unique_lock<std::mutex> lk(cv_m);
#ifdef __INTEL_COMPILER
	// for some reason, the Intel compiler didn't implement std::cv_status in a normal way.
	// for intel 15, it is in the tbb::interface5 namespace.
	// enum cv_status { no_timeout, timeout }; 
    auto stat = cv.wait_for(lk, std::chrono::seconds(1));
	// assume the enum starts at 0?
    if (stat == 0) { break; }; // if we were signalled, exit.
#else
    std::cv_status stat = cv.wait_for(lk, std::chrono::seconds(1));
    if (stat != std::cv_status::timeout) { break; }; // if we were signalled, exit.
#endif

#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau()) {
      TAU_START("ProcData::read_proc: main loop");
    }
#endif
    if (apex_options::use_proc_stat()) {
        // take a reading
        newData = parse_proc_stat();
        periodData = newData->diff(*oldData);
        // save the values
        periodData->sample_values();
        // free the memory
        delete(oldData);
        delete(periodData);
        oldData = newData;
    }
    parse_proc_meminfo(); // some things change, others don't...
    parse_proc_self_status();
    parse_proc_netdev();

#ifdef APEX_HAVE_LM_SENSORS
    mysensors->read_sensors();
#endif

#ifdef APEX_HAVE_TAU
    if (apex_options::use_tau()) {
      TAU_STOP("ProcData::read_proc: main loop");
    }
#endif
  }
#ifdef APEX_HAVE_LM_SENSORS
  delete(mysensors);
#endif

#ifdef APEX_HAVE_TAU
  if (apex_options::use_tau()) {
    TAU_STOP("ProcData::read_proc");
  }
#endif
  delete(oldData);

}

}

#endif // APEX_HAVE_PROC
