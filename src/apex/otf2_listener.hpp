//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex.hpp"
#include "event_listener.hpp"
#include <otf2/otf2.h>
#include <map>
#include <string>
#include <mutex>
#include <chrono>

namespace apex {

    class otf2_listener : public event_listener {
    private:
        void _init(void);
        bool _terminate;
        std::mutex _mutex;
        uint64_t globalOffset;
        static OTF2_TimeStamp get_time( void ) {
            static __thread uint64_t sequence(0);
            return sequence++;
        }
        static OTF2_FlushType pre_flush( void* userData, 
                                         OTF2_FileType fileType, 
                                         OTF2_LocationRef location, 
                                         void* callerData, bool final ) 
        { return OTF2_FLUSH; }
        static OTF2_TimeStamp post_flush( void* userData, 
                                          OTF2_FileType fileType, 
                                          OTF2_LocationRef location ) 
        { return get_time(); }
        static OTF2_FlushCallbacks flush_callbacks;
        void* event_writer(void* arg);
        OTF2_Archive* archive;
        static __thread OTF2_EvtWriter* evt_writer;
        //static __thread OTF2_DefWriter* def_writer;
        OTF2_EvtWriter* getEvtWriter();
        OTF2_DefWriter* getDefWriter(int threadid);
        OTF2_GlobalDefWriter* global_def_writer;
        std::map<task_identifier,uint64_t>& get_region_indices(void) {
            static __thread std::map<task_identifier,uint64_t> region_indices;
            return region_indices;
        }
        std::map<task_identifier,uint64_t>& get_global_region_indices(void) {
            static std::map<task_identifier,uint64_t> region_indices;
            return region_indices;
        }
        uint64_t get_region_index(task_identifier* id) {
            /* first, look in this thread's map */
            std::map<task_identifier,uint64_t>& region_indices = get_region_indices();
            auto tmp = region_indices.find(*id);
            uint64_t region_index = 0;
            if (tmp == region_indices.end()) {
                /* not in the thread's map? look in the global map */
                std::map<task_identifier,uint64_t>& global_region_indices = get_global_region_indices();
                std::unique_lock<std::mutex> lock(_mutex);
                tmp = global_region_indices.find(*id);
                if (tmp == global_region_indices.end()) {
                    /* not in the global map? create it. */
                    region_index = global_region_indices.size();
                    global_region_indices[*id] = region_index;
                } else {
                    region_index = tmp->second;
                }
                lock.unlock();
                /* store the global value in the thread's map */
                region_indices[*id] = region_index;
            } else {
                region_index = tmp->second;
            }
	        return region_index;
        }
        uint64_t get_string_index(const std::string& name) {
  	        static __thread std::map<std::string,uint64_t> string_indices;
  	        static std::map<std::string,uint64_t> global_string_indices;
            /* first, look in this thread's map */
            auto tmp = string_indices.find(name);
            uint64_t string_index = 0;
            if (tmp == string_indices.end()) {
                /* not in the thread's map? look in the global map */
                std::unique_lock<std::mutex> lock(_mutex);
                tmp = global_string_indices.find(name);
                if (tmp == global_string_indices.end()) {
                    string_index = global_string_indices.size();
                    global_string_indices[name] = string_index;
                } else {
                    string_index = tmp->second;
                }
                lock.unlock();
                /* stoer the global value in the thread's map */
                string_indices[name] = string_index;
            } else {
                string_index = tmp->second;
            }
	        return string_index;
        }
        uint64_t get_hostname_index(const std::string& name) {
            /* first, look in the map */
            static std::map<std::string,uint64_t> hostname_indices;
            auto tmp = hostname_indices.find(name);
            uint64_t hostname_index = 0;
            if (tmp == hostname_indices.end()) {
                /* not in the map? create it. */
                hostname_index = hostname_indices.size();// + 1;
                /* store the global value in the thread's map */
                hostname_indices[name] = hostname_index;
            } else {
                hostname_index = tmp->second;
            }
	        return hostname_index;
        }
        static const std::string empty;
        void write_otf2_regions(void);
        void write_my_regions(void);
        void reduce_regions(void);
        void write_region_map(void);
        void write_clock_properties(void);
        void write_host_properties(int rank, int pid, std::string& hostname);
        static const std::string index_filename;
        static const std::string lock_filename_prefix;
        static const std::string region_filename_prefix;
        void create_archive(void);
        int my_saved_node_id;
        std::map<int,int> rank_thread_map;
        std::map<int,int> rank_region_map;
        std::map<std::string,uint64_t> reduced_map;
    public:
        otf2_listener (void);
        ~otf2_listener (void) { };
        void on_startup(startup_event_data &data);
        void on_shutdown(shutdown_event_data &data);
        void on_new_node(node_event_data &data);
        void on_new_thread(new_thread_event_data &data);
        void on_exit_thread(event_data &data);
        bool on_start(task_identifier *id);
        void on_stop(std::shared_ptr<profiler> &p);
        void on_yield(std::shared_ptr<profiler> &p);
        bool on_resume(task_identifier * id);
        void on_sample_value(sample_value_event_data &data);
        void on_new_task(task_identifier * id, void * task_id)
            { APEX_UNUSED(id); APEX_UNUSED(task_id); };
        void on_periodic(periodic_event_data &data)
            { APEX_UNUSED(data); };
        void on_custom_event(custom_event_data &data)
            { APEX_UNUSED(data); };
    };
}

