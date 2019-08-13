//  Copyright (c) 2014-2018 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "perfstubs/perfstubs_api/Tool.h"
#include <stdlib.h>
#include "apex.h"
#include "thread_instance.hpp"

extern "C" {
    // library function declarations
    void perftool_init(void) {
        apex_init("PerfStubs API", 0, 1);
    }
    void perftool_register_thread(void) {
        apex_register_thread("PerfStubs Thread");
    }
    void perftool_exit(void) {
        apex_exit_thread();
    }
    void perftool_dump(void) {
        apex_dump(false);
    }

    // measurement function declarations
    void perftool_timer_start(const char *timer_name) {
        apex_start(APEX_NAME_STRING, (void*)const_cast<char*>(timer_name));
    }
    void perftool_timer_stop(const char *timer_name) {
        apex_stop(apex::thread_instance::instance().get_current_profiler());
    }
    void perftool_static_phase_start(const char *phase_name) {
        apex_start(APEX_NAME_STRING, (void*)const_cast<char*>(phase_name));
    }
    void perftool_static_phase_stop(const char *phase_name) {
        apex_stop(apex::thread_instance::instance().get_current_profiler());
    }
    void perftool_dynamic_phase_start(const char *iteration_prefix,
                                      int iteration_number) {
        std::stringstream ss;
        ss << iteration_prefix << " " << iteration_number;
        apex_start(APEX_NAME_STRING, (void*)const_cast<char*>(ss.str().c_str()));
    }
    void perftool_dynamic_phase_stop(const char *iteration_prefix,
                                     int iteration_number) {
        apex_stop(apex::thread_instance::instance().get_current_profiler());
    }
    void perftool_sample_counter(const char *counter_name, double value) {
        apex_sample_value(counter_name, value);
    }
    void perftool_metadata(const char *name, const char *value) {
        // do nothing
    }

    // data query function declarations
    void perftool_get_timer_data(perftool_timer_data_t *timer_data) {
        memset(timer_data, 0, sizeof(perftool_timer_data_t));
    }
    void perftool_free_timer_data(perftool_timer_data_t *timer_data) {
        if (timer_data == nullptr)
        {
            return;
        }
        if (timer_data->timer_names != nullptr)
        {
            free(timer_data->timer_names);
            timer_data->timer_names = nullptr;
        }
        if (timer_data->metric_names != nullptr)
        {
            free(timer_data->metric_names);
            timer_data->metric_names = nullptr;
        }
        if (timer_data->values != nullptr)
        {
            free(timer_data->values);
            timer_data->values = nullptr;
        }
    }
    void perftool_get_counter_data(perftool_counter_data_t *counter_data) {
        memset(counter_data, 0, sizeof(perftool_counter_data_t));
    }
    void perftool_free_counter_data(perftool_counter_data_t *counter_data) {
        if (counter_data == nullptr)
        {   
            return;
        }
        if (counter_data->counter_names != nullptr)
        {
            free(counter_data->counter_names);
            counter_data->counter_names = nullptr;
        }
        if (counter_data->num_samples != nullptr)
        {
            free(counter_data->num_samples);
            counter_data->num_samples = nullptr;
        }
        if (counter_data->value_total != nullptr)
        {
            free(counter_data->value_total);
            counter_data->value_total = nullptr;
        }
        if (counter_data->value_min != nullptr)
        {
            free(counter_data->value_min);
            counter_data->value_min = nullptr;
        }
        if (counter_data->value_max != nullptr)
        {
            free(counter_data->value_max);
            counter_data->value_max = nullptr;
        }
        if (counter_data->value_sumsqr != nullptr)
        {
            free(counter_data->value_sumsqr);
            counter_data->value_sumsqr = nullptr;
        }
    }
    void perftool_get_metadata(perftool_metadata_t *metadata) {
        memset(metadata, 0, sizeof(perftool_metadata_t));
    }
    void perftool_free_metadata(perftool_metadata_t *metadata) {
        if (metadata == nullptr)
        {
            return;
        }
        if (metadata->names != nullptr)
        {
            free(metadata->names);
            metadata->names = nullptr;
        }
        if (metadata->values != nullptr)
        {
            free(metadata->values);
            metadata->values = nullptr;
        }
    }

}