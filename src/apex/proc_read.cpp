#if defined(APEX_HAVE_PROC)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "proc_read.h"
#include "apex.hpp"
#include <boost/atomic.hpp>

#define COMMAND_LEN 20
#define DATA_SIZE 512

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
    }
  }
  fclose (pFile);
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
  return d;
}

void ProcData::dump(ostream &out) {
  out << "name\tuser\tnice\tsys\tidle\tiowait\tirq\tsoftirq\tsteal\tguest" << endl;
  CPUs::iterator iter;
  long long total;
  double idle_ratio;
  double user_ratio;
  double system_ratio;
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
	  total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle;
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
  long long total;
  double idle_ratio;
  double user_ratio;
  double system_ratio;
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
  long long total;
  double user_ratio;
  for (iter = cpus.begin(); iter != cpus.end(); ++iter) {
    CPUStat* cpu_stat=*iter;
	  if (strcmp(cpu_stat->name, "cpu") == 0) {
	    total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle;
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
}

void ProcData::sample_values(void) {
  long long total;
  double idle_ratio;
  double user_ratio;
  double system_ratio;
  CPUs::iterator iter = cpus.begin();
  CPUStat* cpu_stat=*iter;
  sample_value("CPU User", cpu_stat->user);
  sample_value("CPU Nice", cpu_stat->nice);
  sample_value("CPU System", cpu_stat->system);
  sample_value("CPU Idle", cpu_stat->idle);
  sample_value("CPU I/O Wait", cpu_stat->iowait);
  sample_value("CPU IRQ", cpu_stat->irq);
  sample_value("CPU soft IRQ", cpu_stat->softirq);
  sample_value("CPU Steal", cpu_stat->steal);
  sample_value("CPU Guest", cpu_stat->guest);
  total = cpu_stat->user + cpu_stat->nice + cpu_stat->system + cpu_stat->idle;
  user_ratio = (double)cpu_stat->user / (double)total;
  system_ratio = (double)cpu_stat->system / (double)total;
  idle_ratio = (double)cpu_stat->idle / (double)total;
  sample_value("CPU User Ratio", user_ratio);
  sample_value("CPU System Ratio", system_ratio);
  sample_value("CPU Idle Ratio", idle_ratio);
}

/* This is the main function for the reader thread. */
void ProcData::read_proc(void) {
  ProcData *oldData = parse_proc_stat();
  ProcData *newData = NULL;
  ProcData *periodData = NULL;
  struct timespec tim, tim2;
  tim.tv_sec = 1;
  tim.tv_nsec = 0;
  
  while(!proc_done) {
    // sleep until next time
    nanosleep(&tim , &tim2);
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
}

}

#endif // APEX_HAVE_PROC
