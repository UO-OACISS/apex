//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex.hpp"
#include "event_listener.hpp"
#include <otf2/otf2.h>
#include <otf2/OTF2_Pthread_Locks.h>
#include <map>
#include <string>
#include <mutex>
#include <chrono>

namespace apex {

    class otf2_listener : public event_listener {
    public:
        using string_map_type = std::map<std::string,uint64_t>;
        using region_map_type =  std::map<task_identifier,uint64_t>;
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


        static OTF2_CallbackCode
        otf2_collectives_apex_get_size( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        uint32_t*               size )
        {
            *size = 0;
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_get_rank( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        uint32_t*               rank )
        {
            *rank = apex::__instance()->get_node_id();
            return OTF2_CALLBACK_SUCCESS;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_barrier( void*                   userData,
                                        OTF2_CollectiveContext* commContext )
        {
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_bcast( void*                   userData,
                                    OTF2_CollectiveContext* commContext,
                                    void*                   data,
                                    uint32_t                numberElements,
                                    OTF2_Type               type,
                                    uint32_t                root )
        {
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_gather( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        const void*             inData,
                                        void*                   outData,
                                        uint32_t                numberElements,
                                        OTF2_Type               type,
                                        uint32_t                root )
        {
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_gatherv( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        const void*             inData,
                                        uint32_t                inElements,
                                        void*                   outData,
                                        const uint32_t*         outElements,
                                        OTF2_Type               type,
                                        uint32_t                root )
        {
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_scatter( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        const void*             inData,
                                        void*                   outData,
                                        uint32_t                numberElements,
                                        OTF2_Type               type,
                                        uint32_t                root )
        {
            return OTF2_CALLBACK_ERROR;
        }


        static OTF2_CallbackCode
        otf2_collectives_apex_scatterv( void*                   userData,
                                        OTF2_CollectiveContext* commContext,
                                        const void*             inData,
                                        const uint32_t*         inElements,
                                        void*                   outData,
                                        uint32_t                outElements,
                                        OTF2_Type               type,
                                        uint32_t                root )
        {
            return OTF2_CALLBACK_ERROR;
        }

        static OTF2_FlushCallbacks flush_callbacks;
        static OTF2_CollectiveCallbacks collective_callbacks;
        void* event_writer(void* arg);
        OTF2_Archive* archive;
        static __thread OTF2_EvtWriter* evt_writer;
        static __thread OTF2_DefWriter* def_writer;
        OTF2_GlobalDefWriter* global_def_writer;
        inline region_map_type & get_region_indices(void) {
            static __thread region_map_type region_indices;
            return region_indices;
        }
        inline string_map_type & get_string_indices(void) {
            static __thread string_map_type string_indices;
            return string_indices;
        }
        uint64_t get_region_index(task_identifier* id);
        uint64_t get_string_index(const std::string& name);
        static const std::string empty;
        inline static uint64_t get_location_id() {
            const uint64_t node_id = apex::__instance()->get_node_id();
            const uint64_t thread_id = thread_instance::get_id();
            return (node_id * APEX_MAX_THREADS_PER_LOCALITY) + thread_id;
        }
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
        void on_new_task(new_task_event_data & data);
        void on_destroy_task(destroy_task_event_data & data);
        void on_new_dependency(new_dependency_event_data & data) { APEX_UNUSED(data); };
        void on_satisfy_dependency(satisfy_dependency_event_data & data) { APEX_UNUSED(data); };
        void on_set_task_state(set_task_state_event_data & data) { APEX_UNUSED(data); };
        void on_acquire_data(acquire_data_event_data &data) { APEX_UNUSED(data); };
        void on_release_data(release_data_event_data &data) { APEX_UNUSED(data); };
        void on_new_event(new_event_event_data &data) { APEX_UNUSED(data); };
        void on_destroy_event(destroy_event_event_data &data) { APEX_UNUSED(data); };
        void on_new_data(new_data_event_data &data) { APEX_UNUSED(data); };
        void on_destroy_data(destroy_data_event_data &data) { APEX_UNUSED(data); };
        void on_periodic(periodic_event_data &data)
            { APEX_UNUSED(data); };
        void on_custom_event(custom_event_data &data)
            { APEX_UNUSED(data); };
    };
}

