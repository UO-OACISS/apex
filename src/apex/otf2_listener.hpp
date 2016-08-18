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
        std::mutex _region_mutex;
        std::mutex _string_mutex;
        std::mutex _metric_mutex;
        std::mutex _comm_mutex;
        static uint64_t globalOffset;
        static OTF2_TimeStamp get_time( void ) {
            using namespace std::chrono;
            uint64_t stamp = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
            stamp = stamp - globalOffset;
            return stamp;
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

        static OTF2_CallbackCode my_OTF2GetSize(void *userData, 
            OTF2_CollectiveContext *commContext, 
            uint32_t *size);
        static OTF2_CallbackCode my_OTF2GetRank (void *userData,
            OTF2_CollectiveContext *commContext, 
            uint32_t *rank);
        static OTF2_CallbackCode my_OTF2CreateLocalComm (void *userData,
            OTF2_CollectiveContext **localCommContext, 
            OTF2_CollectiveContext *globalCommContext, 
            uint32_t globalRank, 
            uint32_t globalSize, 
            uint32_t localRank, 
            uint32_t localSize, 
            uint32_t fileNumber, 
            uint32_t numberOfFiles);
        static OTF2_CallbackCode my_OTF2FreeLocalComm (void *userData,
            OTF2_CollectiveContext *localCommContext);
        static OTF2_CallbackCode my_OTF2Barrier (void *userData,
            OTF2_CollectiveContext *commContext);
        static OTF2_CallbackCode my_OTF2Bcast (void *userData,
            OTF2_CollectiveContext *commContext, 
            void *data, 
            uint32_t numberElements,
            OTF2_Type type, 
            uint32_t root);
        static OTF2_CallbackCode my_OTF2Gather (void *userData, 
            OTF2_CollectiveContext *commContext, 
            const void *inData, 
            void *outData,
            uint32_t numberElements, 
            OTF2_Type type, 
            uint32_t root);
        static OTF2_CallbackCode my_OTF2Gatherv (void *userData, 
            OTF2_CollectiveContext *commContext, 
            const void *inData, 
            uint32_t inElements, 
            void *outData, 
            const uint32_t *outElements, 
            OTF2_Type type, uint32_t root);
        static OTF2_CallbackCode my_OTF2Scatter (void *userData, 
            OTF2_CollectiveContext *commContext, 
            const void *inData, 
            void *outData, 
            uint32_t numberElements, 
            OTF2_Type type, 
            uint32_t root);
        static OTF2_CallbackCode my_OTF2Scatterv (void *userData, 
            OTF2_CollectiveContext *commContext, 
            const void *inData, 
            const uint32_t *inElements, 
            void *outData, 
            uint32_t outElements, 
            OTF2_Type type, 
            uint32_t root);
        static void my_OTF2Release (void *userData, 
            OTF2_CollectiveContext *globalCommContext, 
            OTF2_CollectiveContext *localCommContext);
        static OTF2_CollectiveCallbacks * get_collective_callbacks (void);
	    static OTF2_FlushCallbacks flush_callbacks;
        void* event_writer(void* arg);
        OTF2_Archive* archive;
        static __thread OTF2_EvtWriter* evt_writer;
        static OTF2_EvtWriter* comm_evt_writer;
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
                std::unique_lock<std::mutex> lock(_region_mutex);
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
            // thread specific
  	        static __thread std::map<std::string,uint64_t> string_indices;
            // process specific
  	        static std::map<std::string,uint64_t> global_string_indices;
            /* first, look in this thread's map */
            auto tmp = string_indices.find(name);
            uint64_t string_index = 0;
            if (tmp == string_indices.end()) {
                /* not in the thread's map? look in the global map */
                std::unique_lock<std::mutex> lock(_string_mutex);
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
        std::map<std::string,uint64_t>& get_metric_indices(void) {
            static __thread std::map<std::string,uint64_t> metric_indices;
            return metric_indices;
        }
        std::map<std::string,uint64_t>& get_global_metric_indices(void) {
            static std::map<std::string,uint64_t> metric_indices;
            return metric_indices;
        }
        uint64_t get_metric_index(const std::string& name) {
            // thread specific
  	        std::map<std::string,uint64_t>& metric_indices = get_metric_indices();
            /* first, look in this thread's map */
            auto tmp = metric_indices.find(name);
            uint64_t metric_index = 0;
            if (tmp == metric_indices.end()) {
                // process specific
  	            std::map<std::string,uint64_t>& global_metric_indices = get_global_metric_indices();
                /* not in the thread's map? look in the global map */
                std::unique_lock<std::mutex> lock(_metric_mutex);
                tmp = global_metric_indices.find(name);
                if (tmp == global_metric_indices.end()) {
                    metric_index = global_metric_indices.size();
                    global_metric_indices[name] = metric_index;
                } else {
                    metric_index = tmp->second;
                }
                lock.unlock();
                /* stoer the global value in the thread's map */
                metric_indices[name] = metric_index;
            } else {
                metric_index = tmp->second;
            }
	        return metric_index;
        }
        static const std::string empty;
        void write_otf2_regions(void);
        void write_my_regions(void);
        int reduce_regions(void);
        void write_region_map(void);
        void write_otf2_metrics(void);
        void write_my_metrics(void);
        void reduce_metrics(void);
        void write_metric_map(void);
        void write_clock_properties(void);
        void write_host_properties(int rank, int pid, std::string& hostname);
        static const std::string index_filename;
        static const std::string lock_filename_prefix;
        static const std::string region_filename_prefix;
        static const std::string metric_filename_prefix;
        bool create_archive(void);
        bool write_my_node_properties(void);
        static int my_saved_node_id;
        std::map<int,int> rank_thread_map;
        std::map<int,int> rank_region_map;
        std::map<int,int> rank_metric_map;
        std::map<std::string,uint64_t> reduced_region_map;
        std::map<std::string,uint64_t> reduced_metric_map;
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
        void on_send(message_event_data &data);
        void on_recv(message_event_data &data);
    };
}

