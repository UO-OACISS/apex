//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "otf2_listener.hpp"
#include "thread_instance.hpp"
#include <sstream>

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

    otf2_listener::otf2_listener (void) : _terminate(false), global_def_writer(nullptr) {
        flush_callbacks = { 
            .otf2_pre_flush  = otf2_listener::pre_flush, 
            .otf2_post_flush = otf2_listener::post_flush 
        };
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
                OTF2_COMPRESSION_NONE );
        /* set the flush callbacks, basically getting timestamps */
        OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks, NULL );
        /* ? */
        OTF2_Archive_SetSerialCollectiveCallbacks( archive );
        /* ? */
        OTF2_Pthread_Archive_SetLockingCallbacks( archive, NULL );
        /* open the event files for this archive */
        OTF2_Archive_OpenEvtFiles( archive );
        /* get an event writer for this thread */
        evt_writer = OTF2_Archive_GetEvtWriter( archive, thread_instance::get_id() );
        /* get the definition writers so we can record strings */
        def_writer = OTF2_Archive_GetDefWriter( archive, thread_instance::get_id() );
        global_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );
        // add the empty string to the string definitions
        OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(empty), empty.c_str() );
        return;
    }

    void otf2_listener::write_otf2_regions(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        auto region_indices = get_global_region_indices();
        for (auto const &i : region_indices) {
            task_identifier id = i.first;
            uint64_t idx = i.second;
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(id.get_name()), id.get_name().c_str() );
            OTF2_GlobalDefWriter_WriteRegion( global_def_writer,
                    idx /* id */,
                    get_string_index(id.get_name()) /* region name  */,
                    get_string_index(empty) /* alternative name */,
                    get_string_index(empty) /* description */,
                    OTF2_REGION_ROLE_FUNCTION,
                    OTF2_PARADIGM_USER,
                    OTF2_REGION_FLAG_NONE,
                    get_string_index(empty) /* source file */,
                    get_string_index(empty) /* begin lno */,
                    get_string_index(empty) /* end lno */ );
        }
    }

    void otf2_listener::on_shutdown(shutdown_event_data &data) {
        APEX_UNUSED(data);
        write_otf2_regions();
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
            uint64_t ticks_per_second = 1000000;
            using namespace std::chrono;
            uint64_t traceLength = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - this->globalOffset;
            OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
                    ticks_per_second /* 1,000,000,000 ticks per second */,
                    this->globalOffset /* epoch */,
                    traceLength /* length */ );
            /* define some strings */
            stringstream locality;
            locality << "process " << apex::__instance()->get_node_id();
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(locality.str()), locality.str().c_str() );
            char hostname[128];
            gethostname(hostname, sizeof hostname);
            string host(hostname);
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(host), hostname );
            string node("node");
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(node), node.c_str() );
            OTF2_GlobalDefWriter_WriteSystemTreeNode( global_def_writer,
                0, /* System Tree Node ID */
                get_string_index(host), /* host name string ID */
                get_string_index(node), /* class name string ID */
                OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */ );
            OTF2_GlobalDefWriter_WriteLocationGroup( global_def_writer,
                apex::__instance()->get_node_id() /* id */,
                get_string_index(locality.str()) /* name */,
                OTF2_LOCATION_GROUP_TYPE_PROCESS,
                0 /* system tree node ID */ );
            /* write out the thread locations */
            for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
                stringstream thread;
                thread << "thread " << i;
                OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(thread.str()), thread.str().c_str() );
                OTF2_GlobalDefWriter_WriteLocation( global_def_writer, 
                    i /* id */,
                    get_string_index(thread.str()) /* name */,
                    OTF2_LOCATION_TYPE_CPU_THREAD,
                    get_global_region_indices().size() /* number of events */,
                    apex::__instance()->get_node_id() /* location group ID */ );
            }
            OTF2_Archive_Close( archive );
        }
        return;
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        if (!_terminate) {
        }
        return;
    }

    void otf2_listener::on_new_thread(new_thread_event_data &data) {
        if (!_terminate) {
            /* get an event writer for this thread */
            evt_writer = OTF2_Archive_GetEvtWriter( archive, thread_instance::get_id() );
            /* get the definition writer so we can record strings */
            def_writer = OTF2_Archive_GetDefWriter( archive, thread_instance::get_id() );
        }
        return;
    }

    void otf2_listener::on_exit_thread(event_data &data) {
        if (!_terminate) {
            //write_otf2_regions();
            OTF2_Archive_CloseDefWriter( archive, def_writer );
            OTF2_Archive_CloseEvtWriter( archive, evt_writer );
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

}
