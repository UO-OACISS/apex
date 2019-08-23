//  Copyright (c) 2014-2018 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "perfstubs/perfstubs_api/Tool.h"
#include <stdlib.h>
#include "apex.h"
#include "thread_instance.hpp"
#include <mutex>

std::mutex my_mutex;

namespace external {
    namespace ps_implementation {
        class profiler {
            public:
                profiler(const std::string& name) : _name(name) {}
                std::string _name;
                apex_profiler_handle p;
        };
        class counter {
            public:
                counter(const std::string& name) : _name(name) {}
                std::string _name;
        };
    }
}

namespace apex_ex = external::ps_implementation;
std::unordered_map<std::string, apex_ex::profiler*> profilers;
std::unordered_map<std::string, apex_ex::counter*> counters;

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
    void* perftool_timer_create(const char *timer_name) {
        std::string name(timer_name);
        std::lock_guard<std::mutex> guard(my_mutex);
        auto iter = profilers.find(name);
        if (iter == profilers.end()) {
            apex_ex::profiler * p = new apex_ex::profiler(name);
            profilers.insert(std::pair<std::string,apex_ex::profiler*>(name,p));
            return (void*)p;
        }
        return (void*)iter->second;
    }
    void perftool_timer_start(const void *timer) {
        apex_ex::profiler * p = (apex_ex::profiler *)(const_cast<void*>(timer));
        p->p = apex_start(APEX_NAME_STRING, (void*)(p->_name.c_str()));
    }
    void perftool_timer_stop(const void *timer) {
        apex_ex::profiler * p = (apex_ex::profiler *)(const_cast<void*>(timer));
        apex_stop(p->p);
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
    void* perftool_create_counter(const char *counter_name) {
        std::string name(counter_name);
        std::lock_guard<std::mutex> guard(my_mutex);
        auto iter = counters.find(name);
        if (iter == counters.end()) {
            apex_ex::counter * c = new apex_ex::counter(name);
            counters.insert(std::pair<std::string,apex_ex::counter*>(name,c));
            return (void*)c;
        }
        return (void*)iter->second;
    }
    void perftool_sample_counter(const void *counter, double value) {
        apex_ex::counter * c = (apex_ex::counter *)(const_cast<void*>(counter));
        apex_sample_value(c->_name.c_str(), value);
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