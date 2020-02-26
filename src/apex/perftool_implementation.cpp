//  Copyright (c) 2014-2018 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "perfstubs/perfstubs_api/tool.h"
#include <stdlib.h>
#include "apex.h"
#include "thread_instance.hpp"
#include <mutex>
#include <unordered_map>

std::mutex my_mutex;

extern "C" {
    // library function declarations
    void ps_initialize(void) {
        apex_init("PerfStubs API", 0, 1);
    }
    void ps_register_thread(void) {
        apex_register_thread("PerfStubs Thread");
    }
    void ps_finalize(void) {
        apex_exit_thread();
    }
    void ps_dump_data(void) {
        apex_dump(false);
    }

    // measurement function declarations
    void* ps_timer_create(const char *timer_name) {
        return strdup(timer_name);
    }
    void ps_timer_start(const void *timer) {
        apex_start(APEX_NAME_STRING, const_cast<void*>(timer));
    }
    void ps_timer_stop(const void *timer) {
        apex_stop(apex::thread_instance::instance().get_current_profiler());
    }
    void ps_dynamic_phase_start(const char *iteration_prefix,
                                      int iteration_number) {
        std::stringstream ss;
        ss << iteration_prefix << " " << iteration_number;
        apex_start(APEX_NAME_STRING, (void*)const_cast<char*>(ss.str().c_str()));
    }
    void ps_dynamic_phase_stop(const char *iteration_prefix,
                                     int iteration_number) {
        apex_stop(apex::thread_instance::instance().get_current_profiler());
    }
    void* ps_create_counter(const char *counter_name) {
        return (void*)(strdup(counter_name));
    }
    void ps_sample_counter(const void *counter, double value) {
        apex_sample_value((const char *)(counter), value);
    }
    void ps_set_metadata(const char *name, const char *value) {
        // do nothing
    }

    // data query function declarations
    void ps_get_timer_data(ps_tool_timer_data_t *timer_data) {
        memset(timer_data, 0, sizeof(ps_tool_timer_data_t));
    }
    void ps_free_timer_data(ps_tool_timer_data_t *timer_data) {
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
    void ps_get_counter_data(ps_tool_counter_data_t *counter_data) {
        memset(counter_data, 0, sizeof(ps_tool_counter_data_t));
    }
    void ps_free_counter_data(ps_tool_counter_data_t *counter_data) {
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
    void ps_get_metadata(ps_tool_metadata_t *metadata) {
        memset(metadata, 0, sizeof(ps_tool_metadata_t));
    }
    void ps_free_metadata(ps_tool_metadata_t *metadata) {
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