//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex.hpp"
#include "event_listener.hpp"
#include <otf2/otf2.h>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <memory>
#include <mutex>
#include <chrono>
#include "apex_cxx_shared_lock.hpp"
#include "profiler.hpp"
#include "cuda_thread_node.hpp"

namespace apex {

    class otf2_listener : public event_listener {
    private:
        void _init(void);
        bool _terminate;
        bool _initialized;
        std::mutex _region_mutex;
        std::mutex _string_mutex;
        std::mutex _metric_mutex;
        std::mutex _comm_mutex;
        std::mutex _event_set_mutex;
        std::set<uint32_t> _event_threads;
        std::map<uint32_t, std::string> _event_thread_names;
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
            uint64_t stamp = profiler::get_time_ns();
            stamp = stamp - globalOffset;
            return stamp;
        }
        static OTF2_FlushType pre_flush( void* userData,
                                         OTF2_FileType fileType,
                                         OTF2_LocationRef location,
                                         void* callerData, bool final )
        {
            APEX_UNUSED(userData);
            APEX_UNUSED(fileType);
            APEX_UNUSED(location);
            APEX_UNUSED(callerData);
            APEX_UNUSED(final);
            return OTF2_FLUSH;
        }
        static OTF2_TimeStamp post_flush( void* userData,
                                          OTF2_FileType fileType,
                                          OTF2_LocationRef location )
        {
            APEX_UNUSED(userData);
            APEX_UNUSED(fileType);
            APEX_UNUSED(location);
            return get_time();
        }

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
        OTF2_EvtWriter* getEvtWriter(bool create);
        bool event_file_exists (uint32_t threadid);
        OTF2_DefWriter* getDefWriter(uint32_t threadid);
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
        void write_otf2_attributes(void);
        void write_otf2_regions(void);
        std::string write_my_regions(void);
        int reduce_regions(void);
        void write_region_map(std::map<std::string,uint64_t>&
            reduced_region_map);
        void write_otf2_metrics(void);
        std::string write_my_metrics(void);
        void reduce_metrics(void);
        void write_metric_map(std::map<std::string,uint64_t>& reduced_metric_map);
        std::string write_my_threads(void);
        void reduce_threads(void);
        void write_clock_properties(void);
        void write_host_properties(int rank, int pid, std::string& hostname);
        std::string index_filename;
        std::string lock_filename_prefix;
        std::string lock2_filename_prefix;
        std::string lock3_filename_prefix;
        std::string region_filename_prefix;
        std::string metric_filename_prefix;
        std::string thread_filename_prefix;
        bool create_archive(void);
        std::string write_my_node_properties(void);
        static int my_saved_node_id;
        static int my_saved_node_count;
        std::map<int,int> rank_thread_map;
        std::map<int,int> rank_region_map;
        std::map<int,int> rank_metric_map;
        std::map<int,std::map<uint32_t,std::string> > rank_thread_name_map;
        std::map<std::string,uint64_t> reduced_region_map;
        std::map<std::string,uint64_t> reduced_metric_map;
        std::unique_ptr<std::tuple<std::map<int,int>,
            std::map<int,std::string> > >
            reduce_node_properties(std::string&& str);
#if APEX_HAVE_PAPI
        void write_papi_counters(OTF2_EvtWriter* writer, profiler* prof,
            uint64_t stamp, bool is_enter);
#endif
        std::mutex _vthread_mutex;
        std::map<cuda_thread_node, size_t> vthread_map;
        std::map<uint32_t, OTF2_EvtWriter*> vthread_evt_writer_map;
        uint32_t make_vtid (uint32_t device, uint32_t context, uint32_t stream);
        std::map<uint32_t,uint64_t> last_ts;
        uint64_t dropped;
    public:
        otf2_listener (void);
        //~otf2_listener (void) { shutdown_event_data data(my_saved_node_id,0);
        //on_shutdown(data); };
        ~otf2_listener (void) {
            if (dropped > 0) {
                std::cerr << "APEX: Warning! "
                      << dropped << " Aysnchronous Events were delivered out of "
                      << "order by CUDA/CUPTI.\n"
                      << "These events were ignored. Trace may be impcomplete."
                      << std::endl;
            }
            finalize(); };
        void set_node_id(int node_id, int node_count) {
            this->my_saved_node_id = node_id;
            this->my_saved_node_count = node_count;
        }
        void on_startup(startup_event_data &data);
        void on_dump(dump_event_data &data);
        void on_reset(task_identifier * id)
            { APEX_UNUSED(id); };
        void on_shutdown(shutdown_event_data &data);
        void on_new_node(node_event_data &data);
        void on_new_thread(new_thread_event_data &data);
        void on_exit_thread(event_data &data);
        bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
        void on_stop(std::shared_ptr<profiler> &p);
        void on_yield(std::shared_ptr<profiler> &p);
        bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
        void on_sample_value(sample_value_event_data &data);
        void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
            APEX_UNUSED(tt_ptr);
        };
        void on_periodic(periodic_event_data &data)
            { APEX_UNUSED(data); };
        void on_custom_event(custom_event_data &data)
            { APEX_UNUSED(data); };
        void on_send(message_event_data &data);
        void on_recv(message_event_data &data);
        void on_async_event(uint32_t device, uint32_t context,
            uint32_t stream, std::shared_ptr<profiler> &p);

    };
}

