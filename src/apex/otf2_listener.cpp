//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "otf2_listener.hpp"
#include "thread_instance.hpp"
#include <sstream>

#define APEX_OTF2_EXTENSIONS

using namespace std;

namespace apex {
                            

    const std::string otf2_listener::empty("");
    __thread OTF2_EvtWriter* otf2_listener::evt_writer(nullptr);
    __thread OTF2_DefWriter* otf2_listener::def_writer(nullptr);

    OTF2_FlushCallbacks otf2_listener::flush_callbacks =
    {
        .otf2_pre_flush  = pre_flush,
        .otf2_post_flush = post_flush
    };

    OTF2_CollectiveCallbacks otf2_listener::collective_callbacks =
    {
        .otf2_release           = NULL,
        .otf2_get_size          = otf2_collectives_apex_get_size,
        .otf2_get_rank          = otf2_collectives_apex_get_rank,
        .otf2_create_local_comm = NULL,
        .otf2_free_local_comm   = NULL,
        .otf2_barrier           = otf2_collectives_apex_barrier,
        .otf2_bcast             = otf2_collectives_apex_bcast,
        .otf2_gather            = otf2_collectives_apex_gather,
        .otf2_gatherv           = otf2_collectives_apex_gatherv,
        .otf2_scatter           = otf2_collectives_apex_scatter,
        .otf2_scatterv          = otf2_collectives_apex_scatterv
    };
     
    otf2_listener::otf2_listener (void) : _terminate(false), global_def_writer(nullptr) {
        flush_callbacks = { 
            .otf2_pre_flush  = otf2_listener::pre_flush, 
            .otf2_post_flush = otf2_listener::post_flush 
        };
        collective_callbacks = {
            .otf2_release           = NULL,
            .otf2_get_size          = otf2_collectives_apex_get_size,
            .otf2_get_rank          = otf2_collectives_apex_get_rank,
            .otf2_create_local_comm = NULL,
            .otf2_free_local_comm   = NULL,
            .otf2_barrier           = otf2_collectives_apex_barrier,
            .otf2_bcast             = otf2_collectives_apex_bcast,
            .otf2_gather            = otf2_collectives_apex_gather,
            .otf2_gatherv           = otf2_collectives_apex_gatherv,
            .otf2_scatter           = otf2_collectives_apex_scatter,
            .otf2_scatterv          = otf2_collectives_apex_scatterv
        };
    }

    static inline OTF2_RegionRole get_role(apex_task_id_kind_t kind) {
        switch(kind) {
            case APEX_TASK_ID:
                return OTF2_REGION_ROLE_TASK;
            case APEX_EVENT_ID:
                return OTF2_REGION_ROLE_EVENT;
            case APEX_DATA_ID:
                return OTF2_REGION_ROLE_DATABLOCK;
            default:
                return OTF2_REGION_ROLE_UNKNOWN;
        }    
    }

    uint64_t otf2_listener::get_region_index(task_identifier * id) {
#ifdef APEX_DEBUG
        if(id == nullptr) {
            std::cerr << "NULL task id in get_region_index!" << std::endl;
            std::abort();
        }
#endif
        region_map_type & region_indices = get_region_indices();
        auto tmp = region_indices.find(*id);
        uint64_t region_index = 0;
        if (tmp == region_indices.end()) {
            region_index = region_indices.size() + 1;
            region_indices[*id] = region_index;
            uint64_t empty_index = get_string_index(empty);
            const std::string region_name = id->get_name(); 
            OTF2_DefWriter_WriteRegion( def_writer,       
                    region_index /* id */,
                    get_string_index(region_name) /* region name  */,
                    get_string_index(id->internal_name) /* alternative name */,
                    empty_index /* description */,
                    get_role(id->kind),
                    OTF2_PARADIGM_USER,
                    OTF2_REGION_FLAG_NONE,
                    empty_index /* source file */,
                    empty_index /* begin lno */,
                    empty_index /* end lno */ );
        } else {
            region_index = tmp->second;
        }
        return region_index;
    }

    uint64_t otf2_listener::get_string_index(const std::string name) {
        string_map_type & string_indices = get_string_indices();
        auto tmp = string_indices.find(name);
        uint64_t string_index = 0;
        if (tmp == string_indices.end()) {
            string_index = string_indices.size() + 1;
            string_indices[name] = string_index;
            OTF2_DefWriter_WriteString( def_writer, string_index, name.c_str() );
        } else {
            string_index = tmp->second;
        }
        return string_index;
    }

    void otf2_listener::on_startup(startup_event_data &data) {
        /* get a start time for the trace */
        using namespace std::chrono;
        this->globalOffset = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        /* open the OTF2 archive */
        archive = OTF2_Archive_Open( apex_options::otf2_archive_path(),
                apex_options::otf2_archive_name(),
                OTF2_FILEMODE_WRITE,
                1024 * 1024 /* event chunk size */,
                4 * 1024 * 1024 /* def chunk size */,
                OTF2_SUBSTRATE_POSIX,
                OTF2_COMPRESSION_NONE  );
        /* set the flush callbacks, basically getting timestamps */
        OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks, NULL );
        /* ? */
        OTF2_Archive_SetCollectiveCallbacks( archive,
                &collective_callbacks /* callbacks object */,
                NULL /* collective data */,
                NULL /* global comm context */,
                NULL /* local comm context */ );
        /* ? */
        OTF2_Pthread_Archive_SetLockingCallbacks( archive, NULL );
        /* open the event files for this archive */
        OTF2_Archive_OpenEvtFiles( archive );
        /* get an event writer for this thread */
        evt_writer = OTF2_Archive_GetEvtWriter( archive, get_location_id() );
        /* get the definition writers so we can record strings */
        def_writer = OTF2_Archive_GetDefWriter( archive, get_location_id() );
        return;
    }

    void otf2_listener::on_shutdown(shutdown_event_data &data) {
        APEX_UNUSED(data);
        if (!_terminate) {
            _terminate = true;
            /* close event files */
            OTF2_Archive_CloseEvtFiles( archive );
            /* write the clock properties */
            /*
            uint64_t ticks_per_second = (uint64_t)(1.0/profiler::get_cpu_mhz());
            uint64_t globalOffset = profiler::time_point_to_nanoseconds(profiler::get_global_start());
            uint64_t traceLength = profiler::time_point_to_nanoseconds(profiler::get_global_end());
            */
            if(apex::__instance()->get_node_id() == 0) {
                uint64_t ticks_per_second = 1000000;
                using namespace std::chrono;
                uint64_t traceLength = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - this->globalOffset;
                OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
                        ticks_per_second /* 1,000,000,000 ticks per second */,
                        this->globalOffset /* epoch */,
                        traceLength /* length */ );
                /* define some strings */
            }
            OTF2_Archive_Close( archive );
        }
        return;
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        if (!_terminate) {
            if(data.node_id == 0 && global_def_writer == nullptr) {
                global_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );    
            }
            if(def_writer == nullptr) {
                def_writer = OTF2_Archive_GetDefWriter( archive, get_location_id() );
            }
            uint64_t node_id = apex::__instance()->get_node_id();
            stringstream locality;
            locality << "process " << node_id;
            OTF2_DefWriter_WriteString( def_writer, get_string_index(locality.str()), locality.str().c_str() );
            char hostname[128];
            gethostname(hostname, sizeof hostname);
            string host(hostname);
            OTF2_DefWriter_WriteString( def_writer, get_string_index(host), hostname );
            string node("node");
            OTF2_DefWriter_WriteString( def_writer, get_string_index(node), node.c_str() );
            OTF2_DefWriter_WriteSystemTreeNode( def_writer,
                node_id, /* System Tree Node ID */
                get_string_index(host), /* host name string ID */
                get_string_index(node), /* class name string ID */
                OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */ );
            OTF2_DefWriter_WriteLocationGroup( def_writer,
                node_id /* id */,
                get_string_index(locality.str()) /* name */,
                OTF2_LOCATION_GROUP_TYPE_PROCESS,
                node_id /* system tree node ID */ );
        }
        return;
    }

    void otf2_listener::on_new_thread(new_thread_event_data &data) {
        if (!_terminate) {
            uint64_t i = get_location_id();
            /* get an event writer for this thread */
            if(evt_writer == nullptr) {
                evt_writer = OTF2_Archive_GetEvtWriter( archive, i );
            }
            /* get the definition writer so we can record strings */
            if(def_writer == nullptr) {
                def_writer = OTF2_Archive_GetDefWriter( archive, i );
            }
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_MeasurementOnOff( evt_writer,
                nullptr,
                stamp,
                OTF2_MEASUREMENT_ON);
        }
        return;
    }

    void otf2_listener::on_exit_thread(event_data &data) {
        if (!_terminate) {

            // Write location data
            
            uint64_t thread_id = thread_instance::get_id();
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_MeasurementOnOff( evt_writer,
                nullptr,
                stamp,
                OTF2_MEASUREMENT_OFF);
            uint64_t num_events;
            uint64_t node_id = apex::__instance()->get_node_id();
            uint64_t location_id = get_location_id();
            OTF2_EvtWriter_GetNumberOfEvents(evt_writer, &num_events);
            stringstream thread_name;
            thread_name << "thread " << thread_id;
            OTF2_DefWriter_WriteLocation( def_writer,
                location_id /* location id */,
                get_string_index(thread_name.str().c_str()) /* name */,
                OTF2_LOCATION_TYPE_CPU_THREAD /* type */,
                num_events /* # events */,
                node_id);

            // Close local writers

            OTF2_Archive_CloseDefWriter( archive, def_writer );
            def_writer = nullptr;
            OTF2_Archive_CloseEvtWriter( archive, evt_writer );
            evt_writer = nullptr;
        }
        APEX_UNUSED(data);
        return;
    }

    bool otf2_listener::on_start(task_identifier * id) {
        if (!_terminate) {
          /*
            profiler * p = thread_instance::instance().get_current_profiler();
            uint64_t stamp = profiler::time_point_to_nanoseconds(p->start); 
                     */
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_Enter( evt_writer, NULL, 
                stamp,
                get_region_index(id) /* region */ );
        } else {
            return false;
        }
        return true;
    }

    bool otf2_listener::on_resume(task_identifier * id) {
        return on_start(id);
    }

    void otf2_listener::on_stop(std::shared_ptr<profiler> &p) {
        if (!_terminate) {
          /*
            uint64_t stamp = profiler::time_point_to_nanoseconds(p->end);
                     */
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_Leave( evt_writer, NULL, 
                stamp,
                get_region_index(p->task_id) /* region */ );
        }
        return;
    }

    void otf2_listener::on_yield(std::shared_ptr<profiler> &p) {
        on_stop(p);
    }

    void otf2_listener::on_sample_value(sample_value_event_data &data) {
        if (!_terminate) {
        }
        return;
    }

    void otf2_listener::on_new_task(new_task_event_data & data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_TaskCreate( evt_writer, NULL, 
                stamp,
                get_region_index(data.task_id) /* region */ );
#endif
        }
        return;
    }

    void otf2_listener::on_destroy_task(destroy_task_event_data & data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_TaskDestroy( evt_writer, NULL, 
                stamp,
                get_region_index(data.task_id) /* region */ );
#endif
        }
        return;
    }


    void otf2_listener::on_new_dependency(new_dependency_event_data & data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_AddDependence( evt_writer, NULL, 
                stamp,
                get_region_index(data.src), /* region */
                get_region_index(data.dest) /* region */);
#endif
        }
        return;
    }

    void otf2_listener::on_satisfy_dependency(satisfy_dependency_event_data & data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_SatisfyDependence( evt_writer, NULL, 
                stamp,
                get_region_index(data.src), /* region */
                get_region_index(data.dest) /* region */);
#endif
        }
        return;
    }

    void otf2_listener::on_set_task_state(set_task_state_event_data & data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            if(data.state == APEX_TASK_ELIGIBLE) {
                using namespace std::chrono;
                uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
                OTF2_EvtWriter_TaskRunnable( evt_writer, NULL, 
                    stamp,
                    get_region_index(data.task_id) /* region */);
            }
#endif
        }
        return;
    }

    void otf2_listener::on_acquire_data(acquire_data_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_DataAcquire( evt_writer, NULL, 
                stamp,
                get_region_index(data.task_id), /* region */
                get_region_index(data.data_id), /* region */
                data.size);
#endif
        }
        return;
    }

    void otf2_listener::on_release_data(release_data_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_DataRelease( evt_writer, NULL, 
                stamp,
                get_region_index(data.task_id), /* region */
                get_region_index(data.data_id), /* region */
                data.size);
#endif
        }
        return;
    }

    void otf2_listener::on_new_event(new_event_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_EventCreate( evt_writer, NULL, 
                stamp,
                get_region_index(data.event_id) /* region */ );
#endif
        }
        return;
    }

    void otf2_listener::on_destroy_event(destroy_event_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_EventDestroy( evt_writer, NULL, 
                stamp,
                get_region_index(data.event_id) /* region */ );
#endif
        }
        return;
    }

    void otf2_listener::on_new_data(new_data_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_DataCreate( evt_writer, NULL, 
                stamp,
                get_region_index(data.data_id), /* region */
                data.size);
#endif
        }
        return;
    }

    void otf2_listener::on_destroy_data(destroy_data_event_data &data) {
        if (!_terminate) {
#ifdef APEX_OTF2_EXTENSIONS
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            OTF2_EvtWriter_EventDestroy( evt_writer, NULL, 
                stamp,
                get_region_index(data.data_id) /* region */ );
#endif
        }                  
        return;
    }

}
