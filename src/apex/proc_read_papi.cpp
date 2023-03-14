/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "papi.h"
#define MAX_EVENTS_PER_EVENTSET 1024
#include <iterator>

namespace apex {

    //static bool papi_initialized = false;

    struct PAPIStats {
        std::vector<std::string> active_components;
        std::map<std::string, int> eventsets;
        std::map<std::string, std::vector<std::string> > event_names;
        std::map<std::string, std::vector<std::string> > event_units;
        std::map<std::string, std::vector<int> > event_data_types;
        std::map<std::string, std::vector<double> > event_conversions;
    };

    PAPIStats& getStats() {
        static PAPIStats stats;
        return stats;
    }

    typedef union {
        long long ll;
        double fp;
    } event_result_t;

    void initialize_papi_component(const PAPI_component_info_t *comp_info,
        int component_id) {
        int retval = PAPI_OK;
        // get the PAPI component metrics, if any
        std::stringstream tmpstr(apex_options::papi_component_metrics());
        // use stream iterators to copy the stream to the vector as whitespace
        // separated strings
        std::istream_iterator<std::string> tmpstr_it(tmpstr);
        std::istream_iterator<std::string> tmpstr_end;
        std::set<std::string> requested_component_metrics(tmpstr_it, tmpstr_end);

        //printf("Trying %s PAPI component\n", comp_info->name);
        if (comp_info->num_native_events == 0) {
            if (apex_options::use_verbose()) {
                fprintf(stderr, "PAPI %s component found, but ", comp_info->name);
                fprintf(stderr, "no %s events found.\n", comp_info->name);
                if (comp_info->disabled != 0) {
                    fprintf(stderr, "%s.\n", comp_info->disabled_reason);
                }
            }
            return;
        }
        int eventSet = PAPI_NULL;
        retval = PAPI_create_eventset(&eventSet);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error creating PAPI %s eventset.\n", comp_info->name);
            fprintf(stderr, "PAPI error %d: %s\n", retval,
                    PAPI_strerror(retval));
            return;
        }
        int code = PAPI_NATIVE_MASK;
        int event_modifier = PAPI_ENUM_FIRST;
        std::vector<std::string> event_names;
        std::vector<std::string> event_units;
        std::vector<int> event_data_types;
        std::vector<double> event_conversions;
        if (comp_info->num_native_events > MAX_EVENTS_PER_EVENTSET) {
            fprintf(stderr, "WARNING! PAPI component %s has more events than "
                "APEX is configured to support. "
                "Some counters may be missing.\n", comp_info->name);
        }
        for ( int ii=0; ii< std::min(MAX_EVENTS_PER_EVENTSET,
                 comp_info->num_native_events); ii++ ) {
            // get the event
            retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
            event_modifier = PAPI_ENUM_EVENTS;
            if ( retval != PAPI_OK ) {
                fprintf( stderr, "%s %d %s PAPI_enum_cmp_event failed.\n",
                        __FILE__, __LINE__, comp_info->name);
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
                continue;
            }
            // get the event name
            char event_name[PAPI_MAX_STR_LEN];
            retval = PAPI_event_code_to_name( code, event_name );
            if (retval != PAPI_OK) {
                fprintf(stderr, "%s %d %s PAPI_event_code_to_name failed\n", __FILE__,
                        __LINE__, comp_info->name);
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
                continue;
            }
            /* If the user requested some metrics,
             * and the metric is in the list, collect it. */
            if (requested_component_metrics.size() > 0) {
                if (requested_component_metrics.find(event_name) ==
                    requested_component_metrics.end()) {
                    continue;
                }
            } else if (strstr(comp_info->name, "rapl") != NULL) {
                // skip the counter events...
                if (strstr(event_name, "_CNT") != NULL) { continue; }
                // skip the unit conversion events...
                if (strstr(event_name, ":UNITS:") != NULL) { continue; }
            } else if (strcmp(comp_info->name, "cray_pm") == 0) {
                // skip the caps
                if (strstr(event_name, "_CAP") != NULL) { continue; }
            } else if (strcmp(comp_info->name, "lmsensors") == 0) {
                // ignore the values that don't change over time, either.
                if (strstr(event_name, "_max") != NULL) { continue; }
                if (strstr(event_name, "_crit") != NULL) { continue; }
                if (strstr(event_name, "_interval") != NULL) { continue; }
            }
            // get the event info
            PAPI_event_info_t evinfo;
            retval = PAPI_get_event_info(code,&evinfo);
            if (retval != PAPI_OK) {
                fprintf(stderr, "%s %d Error getting %s event info\n", __FILE__,
                        __LINE__, comp_info->name);
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
                event_units.push_back(std::string("J"));
                event_conversions.push_back(1.0e-9);
            } else if(strcmp(unit, "mW") == 0) {
                event_units.push_back(std::string("W"));
                event_conversions.push_back(0.0001);
            } else {
                event_units.push_back(std::string(unit));
                event_conversions.push_back(1.0);
            }
            event_data_types.push_back(evinfo.data_type);
            event_names.push_back(std::string(event_name));

            retval = PAPI_add_event(eventSet, code);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error adding %s event.\n", comp_info->name);
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
                return;
            }
        }
        retval = PAPI_start(eventSet);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error starting PAPI %s eventset.\n", comp_info->name);
            return;
        }
        std::string name(comp_info->name);
        getStats().active_components.push_back(name);
        getStats().eventsets.insert(
            std::pair<std::string,int>(name,eventSet));
        getStats().event_names.insert(
            std::pair<std::string, std::vector<std::string> >(name, event_names));
        getStats().event_units.insert(
            std::pair<std::string, std::vector<std::string> >(name, event_units));
        getStats().event_data_types.insert(
            std::pair<std::string, std::vector<int> >(name, event_data_types));
        getStats().event_conversions.insert(
            std::pair<std::string, std::vector<double> >(name, event_conversions));
    }

    void initialize_papi_events(void) {
        // get the PAPI components
        std::stringstream tmpstr(apex_options::papi_components());
        // use stream iterators to copy the stream to the vector as whitespace
        // separated strings
        std::istream_iterator<std::string> tmpstr_it(tmpstr);
        std::istream_iterator<std::string> tmpstr_end;
        std::set<std::string> requested_components(tmpstr_it, tmpstr_end);
        int num_components = PAPI_num_components();
        const PAPI_component_info_t *comp_info;
        // are there any components?
        for (int component_id = 0 ; component_id < num_components ; component_id++) {
            comp_info = PAPI_get_component_info(component_id);
            if (comp_info == NULL) {
                fprintf(stderr, "PAPI component info unavailable, no power measurements will be done.\n");
                return;
            }
            //printf("Found %s PAPI component\n", comp_info->name);
            if (requested_components.find(comp_info->name) ==
                requested_components.end()) {
                continue;
            }
            //printf("Trying %s PAPI component\n", comp_info->name);
            initialize_papi_component(comp_info, component_id);
        }
    }

    void read_papi_components() {
        PAPIStats& stats = getStats();
        for (auto component : stats.active_components) {
            long long values[MAX_EVENTS_PER_EVENTSET]; // = (long long *)calloc(stats.event_names.size(), sizeof(long long));
            int retval = PAPI_read(stats.eventsets[component], values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI %s eventset.\n", component.c_str());
                fprintf(stderr, "PAPI error %d: %s\n", retval,
                        PAPI_strerror(retval));
            } else {
                for (size_t i = 0 ; i < stats.event_names[component].size() ; i++) {
                    //data->rapl_metrics.push_back(rapl_values[i]);
                	stringstream ss;
                	ss << stats.event_names[component][i];
                	if (stats.event_units[component][i].length() > 0) {
                    	ss << "(" << stats.event_units[component][i] << ")";
                	}
                	if (stats.event_data_types[component][i] == PAPI_DATATYPE_FP64) {
                    	event_result_t tmp;
                    	tmp.ll = values[i];
                    	sample_value(ss.str().c_str(), tmp.fp * stats.event_conversions[component][i]);
                	} else {
                    	double tmp = (double)values[i];
                    	sample_value(ss.str().c_str(), tmp * stats.event_conversions[component][i]);
                	}
                }
            }
        }
    }

} // namespace

