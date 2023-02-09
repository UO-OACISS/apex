/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#if defined(APEX_HAVE_PROC)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "proc_read.h"
#include "apex_api.hpp"
#include "apex.hpp"
#include <sstream>
#include <fstream>
#include <atomic>
#include <unordered_set>
#include <string>
#include <cctype>
#ifdef __MIC__
#include "boost/regex.hpp"
#define REGEX_NAMESPACE boost
#else
#include <regex>
#define REGEX_NAMESPACE std
#endif
#include <set>
#include "utils.hpp"
#include <chrono>
#include <iomanip>

#define COMMAND_LEN 20
#define DATA_SIZE 512

#include "tau_listener.hpp"

#ifdef APEX_HAVE_LM_SENSORS
#include "sensor_data.hpp"
#endif

#ifdef APEX_HAVE_MSR
#include <msr/msr_core.h>
#include <msr/msr_rapl.h>
#endif

#ifdef APEX_WITH_CUDA
#include "apex_nvml.hpp"
#endif

#ifdef APEX_WITH_HIP
#include "apex_rocm_smi.hpp"
#include "hip_profiler.hpp"
#endif

using namespace std;

#if defined(APEX_HAVE_PAPI)
#include "proc_read_papi.cpp"
#endif

#include <dirent.h>

namespace apex {

#ifdef APEX_WITH_HIP
rsmi::monitor * global_rsmi_reader = nullptr;
#endif

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

    /* NB: We have to use C functions for File I/O because the C++ functions
       seem to be caching the data from the virtual filesystem. So it never
       gets refreshed... */
    void read_cray_power(
        std::unordered_map<std::string,std::string>& cray_power_units,
        std::unordered_map<std::string,uint64_t>& cray_power_values) {
        struct dirent *entry = nullptr;
        DIR *dp = nullptr;
        std::set<std::string> skip = { "startup", "version", "raw_scan_hz", "generation", "freshness" };

        dp = opendir("/sys/cray/pm_counters");
        if (dp != nullptr) {
            while ((entry = readdir(dp))) {
                if (entry->d_type == DT_DIR) continue;
                std::string name(entry->d_name);
                if (strstr(entry->d_name,"_cap") != 0) continue;
                if (skip.count(name) != 0) continue;
                uint64_t tmpint;
                std::string tmpstr;
                std::string fname("/sys/cray/pm_counters/");
                fname+=name;
                FILE * fp = fopen(fname.c_str(), "r");
                if (fp != NULL) {
                    char *line = NULL;
                    size_t len = 0;
                    ssize_t nread;
                    while ((nread = getline(&line, &len, fp)) != -1) {
                        std::istringstream iline{line};
                        while (iline >> tmpint >> tmpstr) {
                            cray_power_units[name] = tmpstr;
                            cray_power_values[name] = tmpint;
                            break;
                        }
                    }
                }
                fclose(fp);
            }
        }
        closedir(dp);
        return;
    }


    ProcData* parse_proc_stat(void) {
        if (!apex_options::use_proc_stat()) return nullptr;

        /*  Reading proc/stat as a file  */
        FILE * pFile;
        char line[128];
        char dummy[32];
        pFile = fopen ("/proc/stat","r");
        ProcData* procData = new ProcData();
        if (pFile == nullptr) perror ("Error opening file");
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
                    // softirq 10953997190 0 1380880059 1495447920 1585783785...
                    // ...15525789 0 12 661586214 0 1519806115
                    //sscanf(line, "%s %d\n", dummy, &procData->btime);
                }
                // don't waste time parsing anything but the mean
                if (!apex_options::use_proc_stat_details()) {
                    break;
                }
            }
        }
        fclose (pFile);
#if defined(APEX_HAVE_CRAY_POWER)
        read_cray_power(procData->cray_power_units, procData->cray_power_values);
#endif
#if defined(APEX_HAVE_POWERCAP_POWER)
        procData->package0 = read_package0();
        procData->dram = read_dram();
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
        d->cray_power_units = rhs.cray_power_units;
        /*
        d->cray_power_values = rhs.cray_power_values;
        */
        // We want relative energy values, so take a diff from the last reading.
        for (auto it : cray_power_units) {
            std::string name(it.first);
            std::string unit(it.second);
            uint64_t value{rhs.cray_power_values.find(name)->second};
            if (unit.compare("J") == 0) {
                d->cray_power_values[name] = cray_power_values[name] - value;
            } else {
                d->cray_power_values[name] = value;
            }
        }
#endif
#if defined(APEX_HAVE_POWERCAP_POWER)
        d->package0 = package0 - rhs.package0;
        d->dram = dram - rhs.dram;
#endif
        return d;
    }

    void ProcData::dump(ostream &out) {
        out << "name\tuser\tnice\tsys\tidle\tiowait"
            << "\tirq\tsoftirq\tsteal\tguest" << endl;
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
                total = cpu_stat->user + cpu_stat->nice + cpu_stat->system +
                    cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq +
                    cpu_stat->steal + cpu_stat->guest;
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
        out << "name\tuser\tnice\tsys\tidle\tiowait"
            << "\tirq\tsoftirq\tsteal\tguest" << endl;
    }

    double ProcData::get_cpu_user() {

        CPUs::iterator iter;
        long long total = 0L;
        double user_ratio = 0.0;
        for (iter = cpus.begin(); iter != cpus.end(); ++iter) {
            CPUStat* cpu_stat=*iter;
            if (strcmp(cpu_stat->name, "cpu") == 0) {
                total = cpu_stat->user + cpu_stat->nice + cpu_stat->system +
                    cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq +
                    cpu_stat->steal + cpu_stat->guest;
                user_ratio = (double)cpu_stat->user / (double)total;
                //break;
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

    void ProcData::sample_values(void) {
        double total;
        CPUs::iterator iter = cpus.begin();
        CPUStat* cpu_stat=*iter;
        total = (double)(cpu_stat->user + cpu_stat->nice + cpu_stat->system +
                cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq +
                cpu_stat->steal + cpu_stat->guest);
        total = total * 0.01; // so we have a percentage in the final values
        sample_value("CPU User %",     ((double)(cpu_stat->user))    / total);
        sample_value("CPU Nice %",     ((double)(cpu_stat->nice))    / total);
        sample_value("CPU System %",   ((double)(cpu_stat->system))  / total);
        sample_value("CPU Idle %",     ((double)(cpu_stat->idle))    / total);
        sample_value("CPU I/O Wait %", ((double)(cpu_stat->iowait))  / total);
        sample_value("CPU IRQ %",      ((double)(cpu_stat->irq))     / total);
        sample_value("CPU soft IRQ %", ((double)(cpu_stat->softirq)) / total);
        sample_value("CPU Steal %",    ((double)(cpu_stat->steal))   / total);
        sample_value("CPU Guest %",    ((double)(cpu_stat->guest))   / total);
        if (apex_options::use_proc_stat_details()) {
            iter++;
            int index = 0;
            int width = 1;
            if (cpus.size() > 100) { width = 3; }
            else if (cpus.size() > 10) { width = 2; }
            while (iter != cpus.end()) {
                std::stringstream id;
                id << std::setfill('0');
                CPUStat* cpu_stat=*iter;
                total = (double)(cpu_stat->user + cpu_stat->nice + cpu_stat->system +
                        cpu_stat->idle + cpu_stat->iowait + cpu_stat->irq + cpu_stat->softirq +
                        cpu_stat->steal + cpu_stat->guest);
                double busy = total - cpu_stat->idle;
                total = total * 0.01; // so we have a percentage in the final values
                id << "CPU_" << std::setw(width) << index << " Utilized %";
                sample_value(id.str(), busy / total);
                /*
                id << "CPU_" << std::setw(width) << index << " User %";
                sample_value(id.str(), ((double)(cpu_stat->user)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " Nice %";
                sample_value(id.str(), ((double)(cpu_stat->nice)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " System %";
                sample_value(id.str(), ((double)(cpu_stat->system)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " Idle %";
                sample_value(id.str(), ((double)(cpu_stat->idle)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " I/O Wait %";
                sample_value(id.str(), ((double)(cpu_stat->iowait)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " IRQ %";
                sample_value(id.str(), ((double)(cpu_stat->irq)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " soft IRQ %";
                sample_value(id.str(), ((double)(cpu_stat->softirq)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " Steal %";
                sample_value(id.str(), ((double)(cpu_stat->steal)) / total);
                id.str("");
                id << "CPU_" << std::setw(width) << index << " Guest %";
                sample_value(id.str(), ((double)(cpu_stat->guest)) / total);
                */
                iter++;
                index++;
            }
        }
#if defined(APEX_HAVE_CRAY_POWER)
        for (auto it : cray_power_units) {
            // Ignore zero energy values, they're misleading
            if (cray_power_values[it.first] > 0) {
                std::string name(it.first);
                name += " (";
                name += it.second;
                name += ")";
                sample_value(name, cray_power_values[it.first]);
            }
        }
#endif
#if defined(APEX_HAVE_POWERCAP_POWER)
        sample_value("Package-0 Energy", package0);
        sample_value("DRAM Energy", dram);
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
                tmp = trim(tmp);
                // check for empty line
                if (tmp.size() == 0) continue;
                const REGEX_NAMESPACE::regex separator(":");
                REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(), tmp.end(),
                        separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
                string name = *token;
                if (++token != end) {
                    string value = *token;
                    // check for no value
                    if (value.size() == 0) continue;
                    if (!isdigit(trim(value)[0])) continue;
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

    bool parse_proc_loadavg() {
        if (!apex_options::use_proc_loadavg()) return false;

        FILE *f = fopen("/proc/loadavg", "r");
        if (f) {
            char line[4096] = {0};
            while ( fgets( line, 4096, f)) {
                string tmp(line);
                tmp = trim(tmp);
                // check for empty line
                if (tmp.size() == 0) continue;
                const REGEX_NAMESPACE::regex separator("0");
                REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(), tmp.end(),
                        separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
                string value = *token;
                if (value.size() == 0) continue;
                char* pEnd;
                double d1 = strtod (value.c_str(), &pEnd);
                string cname("1 Minute Load average");
                if (pEnd) { sample_value(cname, d1); }
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
                const REGEX_NAMESPACE::regex separator(":");
                REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(), tmp.end(),
                        separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
                string name = *token++;
                if (token != end) {
                    string value = *token;
                    char* pEnd;
                    double d1 = strtod (value.c_str(), &pEnd);
                    string mname("meminfo:" + name);
                    if (pEnd != NULL) {
                        int len = strlen(pEnd);
                        if( pEnd[len-1] == '\n' )
                            pEnd[len-1] = 0;
                        mname.append(pEnd);
                    }
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
        const std::string vm_prefix("Vm");
        const std::string thread_prefix("Threads");
        const std::string ctx_substr("ctxt_switches");
        if (f) {
            char line[4096] = {0};
            while ( fgets( line, 4096, f)) {
                string tmp(line);
                if (!tmp.compare(0,vm_prefix.size(),vm_prefix)) {
                    const REGEX_NAMESPACE::regex separator(":");
                    REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(),
                            tmp.end(), separator, -1);
                    REGEX_NAMESPACE::sregex_token_iterator end;
                    string name = *token++;
                    if (token != end) {
                        string value = *token;
                        char* pEnd;
                        double d1 = strtod (value.c_str(), &pEnd);
                        string mname("status:" + name);
                        if (pEnd != NULL) {
                        int len = strlen(pEnd);
                            if( pEnd[len-1] == '\n' )
                                pEnd[len-1] = 0;
                            mname.append(pEnd);
                        }
                        if (pEnd) { sample_value(mname, d1); }
                    }
                }
                if (!tmp.compare(0,thread_prefix.size(),thread_prefix)) {
                    const REGEX_NAMESPACE::regex separator(":");
                    REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(),
                            tmp.end(), separator, -1);
                    REGEX_NAMESPACE::sregex_token_iterator end;
                    string name = *token++;
                    if (token != end) {
                        string value = *token;
                        char* pEnd;
                        double d1 = strtod (value.c_str(), &pEnd);
                        string mname("status:" + name);
                        if (pEnd != NULL) {
                        int len = strlen(pEnd);
                            if( pEnd[len-1] == '\n' )
                                pEnd[len-1] = 0;
                            mname.append(pEnd);
                        }
                        if (pEnd) { sample_value(mname, d1); }
                    }
                }
                if (tmp.find(ctx_substr) != tmp.npos) {
                    const REGEX_NAMESPACE::regex separator(":");
                    REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(),
                            tmp.end(), separator, -1);
                    REGEX_NAMESPACE::sregex_token_iterator end;
                    string name = *token++;
                    if (token != end) {
                        string value = *token;
                        char* pEnd;
                        double d1 = strtod (value.c_str(), &pEnd);
                        string mname("status:" + name);
                        if (pEnd != NULL) {
                        int len = strlen(pEnd);
                            if( pEnd[len-1] == '\n' )
                                pEnd[len-1] = 0;
                            mname.append(pEnd);
                        }
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

    bool parse_proc_self_io() {
        if (!apex_options::use_proc_self_io()) return false;
        FILE *f = fopen("/proc/self/io", "r");
        if (f) {
            char line[4096] = {0};
            while ( fgets( line, 4096, f)) {
                string tmp(line);
                const REGEX_NAMESPACE::regex separator(":");
                REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(), tmp.end(),
                        separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
                string name = *token++;
                if (token != end) {
                    string value = *token;
                    char* pEnd;
                    double d1 = strtod (value.c_str(), &pEnd);
                    string mname("io:" + name);
                    if (pEnd) { sample_value(mname, d1); }
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
        FILE *f = fopen("/proc/self/net/dev", "r");
        if (f) {
            char line[4096] = {0};
            char * rc = fgets(line, 4096, f); // skip this line
            if (rc == nullptr) {
                fclose(f);
                return false;
            }
            rc = fgets(line, 4096, f); // skip this line
            if (rc == nullptr) {
                fclose(f);
                return false;
            }
            while (fgets(line, 4096, f)) {
                string outer_tmp(line);
                outer_tmp = trim(outer_tmp);
                const REGEX_NAMESPACE::regex separator("[|:\\s]+");
                REGEX_NAMESPACE::sregex_token_iterator token(outer_tmp.begin(),
                        outer_tmp.end(), separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
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
        for (std::unordered_set<string>::const_iterator it = devices.begin();
                it != devices.end(); it++) {
            // for each device, find out how many sensors there are.
        }
#endif
        return true;
    }

    /* This is the main function for the reader thread. */
    //void* proc_data_reader::read_proc(void * _ptw) {
    void* proc_data_reader::read_proc() {
        in_apex prevent_deadlocks;
        // when tracking memory allocations, ignore these
        in_apex prevent_nonsense;
        //pthread_wrapper* ptw = (pthread_wrapper*)_ptw;
        // make sure APEX knows this is not a worker thread
        thread_instance::instance(false).set_worker(false);
        //ptw->_running = true;
        if (apex_options::pin_apex_threads()) {
            set_thread_affinity();
        }
        /* make sure the profiler_listener has a queue that this
         * thread can push sampled values to */
        apex::async_thread_setup();
        static bool _initialized = false;
        if (!_initialized) {
            initialize_worker_thread_for_tau();
            _initialized = true;
        }
        if (apex_options::use_tau()) {
            tau_listener::Tau_start_wrapper("proc_data_reader::read_proc");
        }
        if (done) { return nullptr; }
#if defined(APEX_HAVE_PAPI)
        initialize_papi_events();
#endif
#ifdef APEX_HAVE_LM_SENSORS
        sensor_data * mysensors = new sensor_data();
#endif
        ProcData *oldData = parse_proc_stat();
#if defined(APEX_HAVE_PAPI)
        read_papi_components();
#endif
        // disabled for now - not sure that it is useful
        parse_proc_cpuinfo(); // do this once, it won't change.
        parse_proc_meminfo(); // some things change, others don't...
        parse_proc_self_status(); // some things change, others don't...
        parse_proc_self_io(); // some things change, others don't...
        parse_proc_loadavg(); // this will change
        parse_proc_netdev();
#ifdef APEX_HAVE_LM_SENSORS
        if (apex_options::use_lm_sensors()) {
            mysensors->read_sensors();
        }
#endif
        ProcData *newData = nullptr;
        ProcData *periodData = nullptr;
#ifdef APEX_WITH_CUDA
        // monitoring option is checked in the constructor
        nvml::monitor * nvml_reader = nullptr;
        if (apex_options::monitor_gpu()) {
            nvml_reader = new nvml::monitor();
            nvml_reader->query();
        }
#endif
#ifdef APEX_WITH_HIP
        rsmi::monitor * rsmi_reader;
        // If PAPI support is lacking, use our own support
        if (apex_options::monitor_gpu()) {
            rsmi_reader = new rsmi::monitor();
            global_rsmi_reader = rsmi_reader;
            rsmi_reader->query();
        }
        rocprofiler::monitor * rocprof_reader;
        // If PAPI support is lacking, use our own support
        if (apex_options::use_hip_profiler()) {
            rocprof_reader = new rocprofiler::monitor();
            rocprof_reader->query();
        }
#endif
        // release the main thread to continue
        while(!done /*&& ptw->wait()*/) {
            //usleep(apex_options::proc_period());
            std::unique_lock<std::mutex> lk(proc_data_reader::cv_m);
            // if we've been interrupted by the main thread, break and exit
            if(cv.wait_for(lk, apex_options::proc_period()*1us) ==
               std::cv_status::no_timeout) {
               break;
            }
            if (done) break;
            if (apex_options::use_tau()) {
                tau_listener::Tau_start_wrapper("proc_data_reader::read_proc: main loop");
            }
            if (apex_options::use_proc_stat()) {
                // take a reading
                newData = parse_proc_stat();
                periodData = newData->diff(*oldData);
                // save the values
                if (done) break; // double-check...
                periodData->sample_values();
                // free the memory
                delete(oldData);
                delete(periodData);
                oldData = newData;
            }
            parse_proc_loadavg();
            parse_proc_meminfo(); // some things change, others don't...
            parse_proc_self_status();
            parse_proc_self_io();
            parse_proc_netdev();
#if defined(APEX_HAVE_PAPI)
            read_papi_components();
#endif

#ifdef APEX_HAVE_LM_SENSORS
            if (apex_options::use_lm_sensors()) {
                mysensors->read_sensors();
            }
#endif
#ifdef APEX_WITH_CUDA
            if (nvml_reader != nullptr) {
                nvml_reader->query();
            }
#endif
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                rsmi_reader->query();
            }
            if (apex_options::use_hip_profiler()) {
                rocprof_reader->query();
            }
#endif
            if (apex_options::use_tau()) {
                tau_listener::Tau_stop_wrapper("proc_data_reader::read_proc: main loop");
            }
        }
#ifdef APEX_HAVE_LM_SENSORS
        delete(mysensors);
#endif

#ifdef APEX_WITH_CUDA
        if (nvml_reader != nullptr) {
            nvml_reader->stop();
        }
#endif
#ifdef APEX_WITH_HIP
        if (apex_options::monitor_gpu()) {
            rsmi_reader->stop();
        }
        if (apex_options::use_hip_profiler()) {
            rocprof_reader->stop();
        }
#endif
        if (apex_options::use_tau()) {
            tau_listener::Tau_stop_wrapper("proc_data_reader::read_proc");
        }
        delete(oldData);
        //ptw->_running = false;
        return nullptr;
    }

#ifdef APEX_HAVE_MSR
    void apex_init_msr(void) {
        int status = init_msr();
        if(status) {
            fprintf(stderr, "Unable to initialize libmsr: %d.\n", status);
            return;
        }
        struct rapl_data * r = nullptr;
        uint64_t * rapl_flags = nullptr;
        status = rapl_init(&r, &rapl_flags);
        if(status < 0) {
            fprintf(stderr, "Unable to initialize rapl component of libmsr: %d.\n",
                    status);
            return;
        }
    }

    void apex_finalize_msr(void) {
        finalize_msr();
    }

    double msr_current_power_high(void) {
        static int initialized = 0;
        static uint64_t * rapl_flags = nullptr;
        static struct rapl_data * r = nullptr;
        static uint64_t sockets = 0;

        if(!initialized) {
            sockets = num_sockets();
            int status = rapl_storage(&r, &rapl_flags);
            if(status) {
                fprintf(stderr, "Error in rapl_storage: %d.\n", status);
                return 0.0;
            }
            initialized = 1;
        }

        poll_rapl_data();
        double watts = 0.0;
        for(int s = 0; s < sockets; ++s) {
            if(r->pkg_watts != nullptr) {
                watts += r->pkg_watts[s];
            }
        }
        return watts;
    }
#endif

    std::string proc_data_reader::get_command_line(void) {
        std::string line;
        std::fstream myfile("/proc/self/cmdline", ios_base::in);
        if (myfile.is_open()) {
            getline (myfile,line);
            myfile.close();
/* From the documentation:
 * This  holds  the complete command line for the process, unless the
 * process is a zombie.  In the latter case, there is nothing in this
 * file: that is, a read on  this  file  will return  0 characters.  The
 * command-line arguments appear in this file as a set of null-separated
 * strings, with a further null byte ('\0') after the last string. */
            // so replace all the nulls with spaces
            std::replace(line.begin(), line.end(), '\0', ' ');
        } else {
            // it wasn't there, so return nothing.
        }
        return line;
    }

std::array<double,2> getAvailableMemory() {
    std::array<double,2> values{0,0};
    /* Get the CPU memory */
        FILE *f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[4096] = {0};
            while ( fgets( line, 4096, f)) {
                string tmp(line);
                const REGEX_NAMESPACE::regex separator(":");
                REGEX_NAMESPACE::sregex_token_iterator token(tmp.begin(), tmp.end(),
                        separator, -1);
                REGEX_NAMESPACE::sregex_token_iterator end;
                string name = *token++;
                if (token != end && name.find("MemFree") != name.npos) {
                    string value = *token;
                    char* pEnd;
                    double d1 = strtod (value.c_str(), &pEnd);
                    if (pEnd) { values[0] =  d1; }
                    break;
                }
            }
            fclose(f);
        }
#ifdef APEX_WITH_HIP
        if (global_rsmi_reader != nullptr) {
            values[1] = global_rsmi_reader->getAvailableMemory();
        }
#endif
    return values;
}

} // namespace

#endif // APEX_HAVE_PROC
