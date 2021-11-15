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

namespace apex {

#if defined(APEX_HAVE_PAPI)

#include "papi.h"

    static bool rapl_initialized = false;
    static bool nvml_initialized = false;
    static bool rocm_initialized = false;
    static bool lms_initialized = false;
    static int rapl_EventSet = PAPI_NULL;
    static int nvml_EventSet = PAPI_NULL;
    static int rocm_EventSet = PAPI_NULL;
    static int lms_EventSet = PAPI_NULL;
    static std::vector<std::string> rapl_event_names;
    static std::vector<std::string> nvml_event_names;
    static std::vector<std::string> rocm_event_names;
    static std::vector<std::string> lms_event_names;
    static std::vector<std::string> rapl_event_units;
    static std::vector<std::string> nvml_event_units;
    static std::vector<std::string> lms_event_units;
    static std::vector<int> nvml_event_codes;
    static std::vector<int> rocm_event_codes;
    static std::vector<int> lms_event_codes;
    static std::vector<int> rapl_event_data_type;
    static std::vector<int> nvml_event_data_type;
    static std::vector<int> lms_event_data_type;
    static std::vector<double> rapl_event_conversion;
    static std::vector<double> nvml_event_conversion;
    static std::vector<double> lms_event_conversion;

    typedef union {
        long long ll;
        double fp;
    } event_result_t;

    void initialize_papi_events(void) {
        // get the PAPI components
        int num_components = PAPI_num_components();
        const PAPI_component_info_t *comp_info;
        int retval = PAPI_OK;
        // are there any components?
        for (int component_id = 0 ; component_id < num_components ; component_id++) {
            comp_info = PAPI_get_component_info(component_id);
            if (comp_info == NULL) {
                fprintf(stderr, "PAPI component info unavailable, no power measurements will be done.\n");
                return;
            }
            // do we have the RAPL components?
            if (strstr(comp_info->name, "rapl")) {
                if (comp_info->num_native_events == 0) {
                    if (apex_options::use_verbose()) {
                        fprintf(stderr, "PAPI RAPL component found, but ");
                        fprintf(stderr, "no RAPL events found.\n");
                        if (comp_info->disabled != 0) {
                            fprintf(stderr, "%s.\n", comp_info->disabled_reason);
                        }
                    }
                } else {
                    rapl_EventSet = PAPI_NULL;
                    retval = PAPI_create_eventset(&rapl_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error creating RAPL PAPI eventset.\n");
                        fprintf(stderr, "PAPI error %d: %s\n", retval,
                                PAPI_strerror(retval));
                        return;
                    }
                    int code = PAPI_NATIVE_MASK;
                    int event_modifier = PAPI_ENUM_FIRST;
                    for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
                        // get the event
                        retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
                        event_modifier = PAPI_ENUM_EVENTS;
                        if ( retval != PAPI_OK ) {
                            fprintf( stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "RAPL PAPI_enum_cmp_event failed.", retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event name
                        char event_name[PAPI_MAX_STR_LEN];
                        retval = PAPI_event_code_to_name( code, event_name );
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s\n", __FILE__,
                                    __LINE__, "RAPL PAPI_event_code_to_name failed");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // skip the counter events...
                        if (strstr(event_name, "_CNT") != NULL) { continue; }
                        // get the event info
                        PAPI_event_info_t evinfo;
                        retval = PAPI_get_event_info(code,&evinfo);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s\n", __FILE__,
                                    __LINE__, "Error getting RAPL event info");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event units
                        char unit[PAPI_MAX_STR_LEN] = {0};
                        strncpy(unit,evinfo.units,PAPI_MAX_STR_LEN);
                        // save the event info
                        //printf("Found event '%s (%s)'\n", event_name, unit);
                        if(strcmp(unit, "nJ") == 0) {
                            rapl_event_units.push_back(std::string("J"));
                            rapl_event_conversion.push_back(1.0e-9);
                        } else {
                            rapl_event_units.push_back(std::string(unit));
                            rapl_event_conversion.push_back(1.0);
                        }
                        rapl_event_data_type.push_back(evinfo.data_type);
                        rapl_event_names.push_back(std::string(event_name));

                        retval = PAPI_add_event(rapl_EventSet, code);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "Error adding RAPL event.\n");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            return;
                        }
                    }
                    retval = PAPI_start(rapl_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error starting PAPI RAPL eventset.\n");
                        return;
                    }
                    rapl_initialized = true;
                }
            }
            // do we have the NVML (cuda) components?
            if (strstr(comp_info->name, "nvml")) {
                if (comp_info->num_native_events == 0) {
                    if (apex_options::use_verbose()) {
                        fprintf(stderr, "PAPI NVML component found, but ");
                        fprintf(stderr, "no NVML events found.\n");
                        if (comp_info->disabled != 0) {
                            fprintf(stderr, "%s.\n", comp_info->disabled_reason);
                        }
                    }
                } else {
                    nvml_EventSet = PAPI_NULL;
                    retval = PAPI_create_eventset(&nvml_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error creating NVML PAPI eventset.\n");
                        fprintf(stderr, "PAPI error %d: %s\n", retval,
                                PAPI_strerror(retval));
                        return;
                    }
                    int code = PAPI_NATIVE_MASK;
                    int event_modifier = PAPI_ENUM_FIRST;
                    for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
                        // get the event
                        retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
                        event_modifier = PAPI_ENUM_EVENTS;
                        if ( retval != PAPI_OK ) {
                            fprintf( stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "NVML PAPI_event_code_to_name", retval );
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event name
                        char event_name[PAPI_MAX_STR_LEN];
                        retval = PAPI_event_code_to_name( code, event_name );
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "NVML Error getting event name\n",retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event info
                        PAPI_event_info_t evinfo;
                        retval = PAPI_get_event_info(code,&evinfo);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "Error getting NVML event info\n",retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event units
                        char unit[PAPI_MAX_STR_LEN] = {0};
                        strncpy(unit,evinfo.units,PAPI_MAX_STR_LEN);
                        // save the event info
                        //printf("Found event '%s (%s)'\n", event_name, unit);
                        nvml_event_codes.push_back(code);
                        if(strcmp(unit, "mW") == 0) {
                            nvml_event_units.push_back(std::string("W"));
                            nvml_event_conversion.push_back(0.0001);
                        } else {
                            nvml_event_units.push_back(std::string(unit));
                            nvml_event_conversion.push_back(1.0);
                        }
                        nvml_event_data_type.push_back(evinfo.data_type);
                        nvml_event_names.push_back(std::string(event_name));
                        retval = PAPI_add_event(nvml_EventSet, code);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "Error adding NVML event.\n");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            return;
                        }
                    }
                    retval = PAPI_start(nvml_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error starting PAPI NVML eventset.\n");
                        fprintf(stderr, "PAPI error %d: %s\n", retval,
                                PAPI_strerror(retval));
                        return;
                    }
                    nvml_initialized = true;
                }
            }
            // do we have the ROCm (hip/rocm) components?
            if (strstr(comp_info->name, "rocm") && !rocm_initialized) {
                printf("Found rocm PAPI component\n");
                if (comp_info->num_native_events == 0) {
                    if (apex_options::use_verbose()) {
                        fprintf(stderr, "PAPI ROCm component found, but ");
                        fprintf(stderr, "no ROCm events found.\n");
                        if (comp_info->disabled != 0) {
                            fprintf(stderr, "%s.\n", comp_info->disabled_reason);
                        }
                    }
                } else {
                    /* Get the list of requested metrics from the options */
                    std::vector<std::string> rocm_metric_names;
                    std::stringstream metric_ss(apex_options::rocprof_metrics());
                    while(metric_ss.good()) {
                        std::string metric;
                        getline(metric_ss, metric, ','); // tokenize by comma
                        rocm_metric_names.push_back(metric);
                    }

                    rocm_EventSet = PAPI_NULL;
                    retval = PAPI_create_eventset(&rocm_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error creating ROCm PAPI eventset.\n");
                        fprintf(stderr, "PAPI error %d: %s\n", retval,
                                PAPI_strerror(retval));
                        return;
                    }
                    int code = PAPI_NATIVE_MASK;
                    int event_modifier = PAPI_ENUM_FIRST;
                    for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
                        // get the event
                        retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
                        event_modifier = PAPI_ENUM_EVENTS;
                        if ( retval != PAPI_OK ) {
                            fprintf( stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "ROCm PAPI_event_code_to_name", retval );
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event name
                        char event_name[PAPI_MAX_STR_LEN];
                        retval = PAPI_event_code_to_name( code, event_name );
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "ROCm Error getting event name\n",retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // is the event in the requested list?
                        bool found = false;
                        for (auto mn : rocm_metric_names) {
                            if (std::string(event_name).find("device=0") != std::string::npos) {
                                if (std::string(event_name).find(mn) != std::string::npos) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        // if the user doesn't want this metric, don't add it
                        if (!found) { continue; }
                        rocm_event_codes.push_back(code);
                        rocm_event_names.push_back(std::string(event_name));
                        retval = PAPI_add_event(rocm_EventSet, code);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "Error adding ROCm event.\n");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            return;
                        }
                    }
                    if (rocm_event_codes.size() > 0) {
                        retval = PAPI_start(rocm_EventSet);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "Error starting PAPI ROCm eventset.\n");
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            return;
                        }
                        rocm_initialized = true;
                    }
                }
            }
            if (strstr(comp_info->name, "lmsensors")) {
                if (comp_info->num_native_events == 0) {
                    if (apex_options::use_verbose()) {
                        fprintf(stderr, "PAPI lmsensors component found, but ");
                        fprintf(stderr, "no lmsensors events found.\n");
                        if (comp_info->disabled != 0) {
                            fprintf(stderr, "%s.\n", comp_info->disabled_reason);
                        }
                    }
                } else {
                    lms_EventSet = PAPI_NULL;
                    retval = PAPI_create_eventset(&lms_EventSet);
                    if (retval != PAPI_OK) {
                        fprintf(stderr, "Error creating PAPI lmsensors eventset.\n");
                        fprintf(stderr, "PAPI error %d: %s\n", retval,
                                PAPI_strerror(retval));
                        return;
                    }
                    int code = PAPI_NATIVE_MASK;
                    int event_modifier = PAPI_ENUM_FIRST;
                    for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
                        // get the event
                        retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
                        event_modifier = PAPI_ENUM_EVENTS;
                        if ( retval != PAPI_OK ) {
                            fprintf( stderr, "%s %d %s %d\n",
                                    __FILE__, __LINE__, "lmsensors PAPI_event_code_to_name", retval );
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                        }
                        // get the event name
                        char event_name[PAPI_MAX_STR_LEN];
                        retval = PAPI_event_code_to_name( code, event_name );
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s %d\n", __FILE__,
                                    __LINE__, "Error getting lmsensors event name\n",retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                            continue;
                        }
                        // get the event info
                        PAPI_event_info_t evinfo;
                        retval = PAPI_get_event_info(code,&evinfo);
                        if (retval != PAPI_OK) {
                            fprintf(stderr, "%s %d %s %d\n",
                                    __FILE__, __LINE__, "Error getting lmsensors event info\n",retval);
                            fprintf(stderr, "PAPI error %d: %s\n", retval,
                                    PAPI_strerror(retval));
                        }
                        // get the event units
                        char unit[PAPI_MAX_STR_LEN] = {0};
                        strncpy(unit,evinfo.units,PAPI_MAX_STR_LEN);
                        // add all LMSensors events!  Except the "Core" specific ones.  Can be too many.
                        //char *Core_sub = strstr(event_name, ".Core ");
                        // ignore the values that don't change over time, either.
                        char *max_sub = strstr(event_name, "_max");
                        char *crit_sub = strstr(event_name, "_crit");
                        char *interval_sub = strstr(event_name, "_interval");
                        if (max_sub == NULL &&
                                crit_sub == NULL && interval_sub == NULL) {
                            // save the event info
                            //printf("Found event '%s (%s)'\n", event_name, unit);
                            lms_event_codes.push_back(code);
                            lms_event_units.push_back(std::string(unit));
                            lms_event_data_type.push_back(evinfo.data_type);
                            lms_event_names.push_back(std::string(event_name));
                            retval = PAPI_add_event(lms_EventSet, code);
                            if (retval != PAPI_OK) {
                                fprintf(stderr, "Error adding lmsensors event.\n");
                                fprintf(stderr, "PAPI error %d: %s\n", retval,
                                        PAPI_strerror(retval));
                                continue;
                            }
                        }
                    }
                }
                retval = PAPI_start(lms_EventSet);
                if (retval != PAPI_OK) {
                    fprintf(stderr, "Error starting PAPI lmsensors eventset.\n");
                    fprintf(stderr, "PAPI error %d: %s\n", retval,
                            PAPI_strerror(retval));
                    return;
                }
                lms_initialized = true;
            }
        }
    }

    void read_papi_components(ProcData * data) {
        if (rapl_initialized) {
            long long * rapl_values = (long long *)calloc(rapl_event_names.size(), sizeof(long long));
            int retval = PAPI_read(rapl_EventSet, rapl_values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI RAPL eventset.\n");
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
            } else {
                for (size_t i = 0 ; i < rapl_event_names.size() ; i++) {
                    data->rapl_metrics.push_back(rapl_values[i]);
                }
            }
            free(rapl_values);
        }
        if (nvml_initialized) {
            long long * nvml_values = (long long *)calloc(nvml_event_names.size(), sizeof(long long));
            int retval = PAPI_read(nvml_EventSet, nvml_values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI NVML eventset.\n");
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
            } else {
                for (size_t i = 0 ; i < nvml_event_names.size() ; i++) {
                    data->nvml_metrics.push_back(nvml_values[i]);
                }
            }
            free(nvml_values);
        }
        if (rocm_initialized) {
            long long * rocm_values = (long long *)calloc(rocm_event_names.size(), sizeof(long long));
            int retval = PAPI_read(rocm_EventSet, rocm_values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI ROCm eventset.\n");
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
            } else {
                for (size_t i = 0 ; i < rocm_event_names.size() ; i++) {
                    data->rocm_metrics.push_back(rocm_values[i]);
                }
            }
            free(rocm_values);
        }
        if (lms_initialized) {
            long long * lms_values = (long long *)calloc(lms_event_names.size(), sizeof(long long));
            int retval = PAPI_read(lms_EventSet, lms_values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI lmsensors eventset.\n");
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
            } else {
                for (size_t i = 0 ; i < lms_event_names.size() ; i++) {
                    data->lms_metrics.push_back(lms_values[i]);
                }
            }
            free(lms_values);
        }
        return;
    }

#endif // defined(APEX_HAVE_PAPI)

    std::atomic<bool> proc_data_reader::done(false);


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
        procData->power = read_power();
        procData->power_cap = read_power_cap();
        procData->energy = read_energy();
        procData->freshness = read_freshness();
        procData->generation = read_generation();
#endif
#if defined(APEX_HAVE_POWERCAP_POWER)
        procData->package0 = read_package0();
        procData->dram = read_dram();
#endif
#if defined(APEX_HAVE_PAPI)
        read_papi_components(procData);
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
#if defined(APEX_HAVE_POWERCAP_POWER)
        d->package0 = package0 - rhs.package0;
        d->dram = dram - rhs.dram;
#endif
#if defined(APEX_HAVE_PAPI)
        if (rapl_initialized) {
            // reading might have failed, so only copy if there's data
            for (size_t i = 0 ; i < rapl_metrics.size() ; i++) {
                d->rapl_metrics.push_back(rapl_metrics[i]);
            }
        }
        if (nvml_initialized) {
            // reading might have failed, so only copy if there's data
            for (size_t i = 0 ; i < nvml_metrics.size() ; i++) {
                d->nvml_metrics.push_back(nvml_metrics[i]);
            }
        }
        if (rocm_initialized) {
            // reading might have failed, so only copy if there's data
            for (size_t i = 0 ; i < rocm_metrics.size() ; i++) {
                d->rocm_metrics.push_back(rocm_metrics[i]);
            }
        }
        if (lms_initialized) {
            // reading might have failed, so only copy if there's data
            for (size_t i = 0 ; i < lms_metrics.size() ; i++) {
                d->lms_metrics.push_back(lms_metrics[i]);
            }
        }
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
        sample_value("Power", power);
        sample_value("Power Cap", power_cap);
        sample_value("Energy", energy);
        sample_value("Freshness", freshness);
        sample_value("Generation", generation);
#endif
#if defined(APEX_HAVE_POWERCAP_POWER)
        sample_value("Package-0 Energy", package0);
        sample_value("DRAM Energy", dram);
#endif
#if defined(APEX_HAVE_PAPI)
        if (rapl_initialized) {
            // reading might have failed, so only iterate over the data
            for (size_t i = 0 ; i < rapl_metrics.size() ; i++) {
                stringstream ss;
                ss << rapl_event_names[i];
                if (rapl_event_units[i].length() > 0) {
                    ss << "(" << rapl_event_units[i] << ")";
                }
                if (rapl_event_names[i].find("ENERGY") == string::npos &&
                        rapl_event_data_type[i] == PAPI_DATATYPE_FP64) {
                    event_result_t tmp;
                    tmp.ll = rapl_metrics[i];
                    sample_value(ss.str().c_str(), tmp.fp * rapl_event_conversion[i]);
                } else {
                    double tmp = (double)rapl_metrics[i];
                    sample_value(ss.str().c_str(), tmp * rapl_event_conversion[i]);
                }
            }
        }
        if (nvml_initialized) {
            // reading might have failed, so only iterate over the data
            for (size_t i = 0 ; i < nvml_metrics.size() ; i++) {
                stringstream ss;
                ss << nvml_event_names[i];
                if (nvml_event_units[i].length() > 0) {
                    ss << "(" << nvml_event_units[i] << ")";
                }
                if (nvml_event_data_type[i] == PAPI_DATATYPE_FP64) {
                    event_result_t tmp;
                    tmp.ll = nvml_metrics[i];
                    sample_value(ss.str().c_str(), tmp.fp * nvml_event_conversion[i]);
                } else {
                    double tmp = nvml_metrics[i];
                    sample_value(ss.str().c_str(), tmp * nvml_event_conversion[i]);
                }
            }
        }
        if (rocm_initialized) {
            // reading might have failed, so only iterate over the data
            for (size_t i = 0 ; i < rocm_metrics.size() ; i++) {
                stringstream ss;
                ss << rocm_event_names[i];
                double tmp = rocm_metrics[i];
                sample_value(ss.str().c_str(), tmp);
            }
        }
        if (lms_initialized) {
            // reading might have failed, so only iterate over the data
            for (size_t i = 0 ; i < lms_metrics.size() ; i++) {
                // PAPI scales LM sensor data by 1000,
                // because it doesn't have floating point values..
                sample_value(lms_event_names[i].c_str(), (double)lms_metrics[i]/1000.0);
            }
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
    void* proc_data_reader::read_proc(void * _ptw) {
        in_apex prevent_deadlocks;
        // when tracking memory allocations, ignore these
        in_apex prevent_nonsense;
        pthread_wrapper* ptw = (pthread_wrapper*)_ptw;
        // make sure APEX knows this is not a worker thread
        thread_instance::instance(false).set_worker(false);
        ptw->_running = true;
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
        nvml::monitor nvml_reader;
        nvml_reader.query();
#endif
#ifdef APEX_WITH_HIP
        // monitoring option is checked in the constructor
        rsmi::monitor rsmi_reader;
        rsmi_reader.query();
        //rocprofiler::monitor rocprof_reader;
        //rocprof_reader.query();
#endif
        while(ptw->wait()) {
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

#ifdef APEX_HAVE_LM_SENSORS
            if (apex_options::use_lm_sensors()) {
                mysensors->read_sensors();
            }
#endif
#ifdef APEX_WITH_CUDA
            nvml_reader.query();
#endif
#ifdef APEX_WITH_HIP
            rsmi_reader.query();
            //rocprof_reader.query();
#endif
            if (apex_options::use_tau()) {
                tau_listener::Tau_stop_wrapper("proc_data_reader::read_proc: main loop");
            }
        }
#ifdef APEX_HAVE_LM_SENSORS
        delete(mysensors);
#endif

#ifdef APEX_WITH_HIP
        //rocprof_reader.stop();
#endif
        if (apex_options::use_tau()) {
            tau_listener::Tau_stop_wrapper("proc_data_reader::read_proc");
        }
        delete(oldData);
        ptw->_running = false;
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

} // namespace

#endif // APEX_HAVE_PROC
