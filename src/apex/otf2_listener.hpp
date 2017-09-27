//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex.hpp"
#include "event_listener.hpp"
#include <otf2/otf2.h>
#include <map>
#include <unordered_set>
#include <string>
#include <mutex>
#include <chrono>
#include "apex_cxx_shared_lock.hpp"

namespace apex {

    class otf2_listener : public event_listener {
    private:
        void _init(void);
        bool _terminate;
        std::mutex _region_mutex;
        std::mutex _string_mutex;
        std::mutex _metric_mutex;
        std::mutex _comm_mutex;
        std::mutex _event_set_mutex;
        std::unordered_set<int> _event_threads;
        /* this is a reader/writer lock. Don't close the archive
         * if other threads are writing to it. but allow concurrent
         * access from the writer threads. */
        shared_mutex_type _archive_mutex;
        /* The global offset is referenced from the get_time static function,
         * so it needs to be static itself. */
        static uint64_t globalOffset;
        /* All OTF2 callback functions have to be declared static, so that they 
         * can be registered with the OTF2 library */
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
        OTF2_EvtWriter* comm_evt_writer;
        //APEX_NATIVE_TLS OTF2_DefWriter* def_writer;
        OTF2_EvtWriter* getEvtWriter();
        bool event_file_exists (int threadid);
        OTF2_DefWriter* getDefWriter(int threadid);
        OTF2_GlobalDefWriter* global_def_writer;
        std::map<task_identifier,uint64_t>& get_region_indices(void);
        std::map<std::string,uint64_t> global_string_indices;
        std::map<std::string,uint64_t>& get_string_indices(void);
        std::map<task_identifier,uint64_t> global_region_indices;
        std::map<std::string,uint64_t> hostname_indices;
        uint64_t get_region_index(task_identifier* id);
        uint64_t get_string_index(const std::string& name);
        uint64_t get_hostname_index(const std::string& name);
        std::map<std::string,uint64_t> global_metric_indices;
        std::map<std::string,uint64_t>& get_metric_indices(void);
        uint64_t get_metric_index(const std::string& name);
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
        std::string index_filename;
        std::string lock_filename_prefix;
        std::string region_filename_prefix;
        std::string metric_filename_prefix;
        bool create_archive(void);
        bool write_my_node_properties(void);
        static int my_saved_node_id;
        static int my_saved_node_count;
        std::map<int,int> rank_thread_map;
        std::map<int,int> rank_region_map;
        std::map<int,int> rank_metric_map;
        std::map<std::string,uint64_t> reduced_region_map;
        std::map<std::string,uint64_t> reduced_metric_map;
    public:
        otf2_listener (void);
        //~otf2_listener (void) { shutdown_event_data data(my_saved_node_id,0); on_shutdown(data); };
        ~otf2_listener (void) { finalize(); };
        void on_startup(startup_event_data &data);
        void on_dump(dump_event_data &data);
        void on_reset(task_identifier * id) 
            { APEX_UNUSED(id); };
        void on_shutdown(shutdown_event_data &data);
        void on_new_node(node_event_data &data);
        void on_new_thread(new_thread_event_data &data);
        void on_exit_thread(event_data &data);
        bool on_start(task_identifier *id);
        void on_stop(std::shared_ptr<profiler> &p);
        void on_yield(std::shared_ptr<profiler> &p);
        bool on_resume(task_identifier * id);
        void on_sample_value(sample_value_event_data &data);
        void on_new_task(task_identifier * id, uint64_t task_id)
            { APEX_UNUSED(id); APEX_UNUSED(task_id); };
        void on_periodic(periodic_event_data &data)
            { APEX_UNUSED(data); };
        void on_custom_event(custom_event_data &data)
            { APEX_UNUSED(data); };
        void on_send(message_event_data &data);
        void on_recv(message_event_data &data);
    };
}

