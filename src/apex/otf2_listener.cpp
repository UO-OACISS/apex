//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "otf2_listener.hpp"
#include "thread_instance.hpp"
#include <sstream>
#include <ostream>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

namespace apex {

    const std::string otf2_listener::empty("");
    __thread OTF2_EvtWriter* otf2_listener::evt_writer(nullptr);
    //__thread OTF2_DefWriter* otf2_listener::def_writer(nullptr);
    const std::string otf2_listener::index_filename("./.max_locality.txt");
    const std::string otf2_listener::region_filename_prefix("./.regions.");
    const std::string otf2_listener::lock_filename_prefix("./.regions.lock.");

    OTF2_EvtWriter* otf2_listener::getEvtWriter(void) {
      if (evt_writer == nullptr) {
        uint64_t my_node_id = apex::__instance()->get_node_id();
        my_node_id = (my_node_id << 32) + thread_instance::get_id();
        evt_writer = OTF2_Archive_GetEvtWriter( archive, my_node_id );
      }
      return evt_writer;
    }

    OTF2_DefWriter* otf2_listener::getDefWriter(int threadid) {
        uint64_t my_node_id = my_saved_node_id;
        my_node_id = (my_node_id << 32) + threadid;
        OTF2_DefWriter* def_writer = OTF2_Archive_GetDefWriter( archive, my_node_id );
        return def_writer;
    }

    OTF2_FlushCallbacks otf2_listener::flush_callbacks =
    {
        .otf2_pre_flush  = pre_flush,
        .otf2_post_flush = post_flush
    };

    otf2_listener::otf2_listener (void) : _terminate(false), global_def_writer(nullptr), my_saved_node_id(0) {
        flush_callbacks = { 
            .otf2_pre_flush  = otf2_listener::pre_flush, 
            .otf2_post_flush = otf2_listener::post_flush 
        };
    }

    void otf2_listener::create_archive(void) {
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
        // add the empty string to the string definitions
        get_string_index(empty);

        /* set up the event unification index file */
        struct stat buffer;   
        if (stat (index_filename.c_str(), &buffer) == 0) { 
            struct tm *timeinfo = localtime(&buffer.st_mtime);
            time_t filetime = mktime(timeinfo);
            time_t nowish;
            time(&nowish);
            double seconds = difftime(nowish, filetime);
            /* if the file exists, was it recently created? */
            if (seconds > 10) {
                /* create the file */
                ofstream indexfile(index_filename, ios::out | ios::trunc );
                indexfile.close();
            }
        } else {
          /* create the file */
            ofstream indexfile(index_filename, ios::out | ios::trunc );
            indexfile.close();
        }
        return;
    }

    void otf2_listener::write_otf2_regions(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        //auto region_indices = get_global_region_indices();
        for (auto const &i : reduced_map) {
            //task_identifier id = i.first;
            string id = i.first;
            uint64_t idx = i.second;
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(id), id.c_str() );
            OTF2_GlobalDefWriter_WriteRegion( global_def_writer,
                    idx /* id */,
                    get_string_index(id) /* region name  */,
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

    void otf2_listener::write_my_regions(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        // create my lock file.
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // open my region file
        ostringstream region_filename;
        region_filename << region_filename_prefix << my_saved_node_id;
        ofstream region_file(region_filename.str(), ios::out | ios::trunc );
        // first, output our number of threads.
        region_file << thread_instance::get_num_threads() << endl;
        // then iterate over the regions and write them out.
        auto region_indices = get_global_region_indices();
        for (auto const &i : region_indices) {
            task_identifier id = i.first;
            uint64_t idx = i.second;
            //region_file << id.get_name() << "\t" << idx << endl;
            region_file << id.get_name() << endl;
        }
        // close the region file
        region_file.close();
        // delete the lock file, so rank 0 can read our data.
        std::remove(lock_filename.str().c_str());
    }

    void otf2_listener::reduce_regions(void) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(my_lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // iterate over my region map, and build a map of strings to ids
        auto region_indices = get_global_region_indices();
        // save my number of regions
        rank_region_map[0] = region_indices.size();
        for (auto const &i : region_indices) {
            task_identifier id = i.first;
            uint64_t idx = i.second;
            reduced_map[id.get_name()] = idx;
        }
        // iterate over the other ranks in the index file
        std::string indexline;
        std::ifstream index_file(index_filename);
        int rank, pid;
        std::string hostname;
        while (std::getline(index_file, indexline)) {
            istringstream ss(indexline);
            ss >> rank >> pid >> hostname;
            // skip myself
            if (rank == 0) continue;
            rank_region_map[rank] = 0;
            struct stat buffer;   
            // wait on the map file to exist
            ostringstream region_filename;
            region_filename << region_filename_prefix << rank;
            while (stat (region_filename.str().c_str(), &buffer) != 0) {}
            // wait for the lock file to not exist
            ostringstream lock_filename;
            lock_filename << lock_filename_prefix << rank;
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            // get the number of threads from that rank
            std::string region_line;
            std::ifstream region_file(region_filename.str());
            std::getline(region_file, region_line);
            std::string::size_type sz;   // alias of size_t
            rank_thread_map[rank] = std::stoi(region_line,&sz);
            // read the map from that rank
            while (std::getline(region_file, region_line)) {
                rank_region_map[rank] = rank_region_map[rank] + 1;
                // trim the newline
                region_line.erase(std::remove(region_line.begin(), region_line.end(), '\n'), region_line.end());
                if (reduced_map.find(region_line) == reduced_map.end()) {
                    uint64_t idx = reduced_map.size();
                    reduced_map[region_line] = idx;
                }
            }
            // close the region file
            region_file.close();
            // remove that rank's map
            std::remove(region_filename.str().c_str());
        }
        // open my region file
        ostringstream region_filename;
        region_filename << region_filename_prefix << my_saved_node_id;
        ofstream region_file(region_filename.str(), ios::out | ios::trunc );
        // copy the reduced map to a pair, so we can sort by value
        std::vector<std::pair<std::string, int>> pairs;
        for (auto const &i : reduced_map) {
            pairs.push_back(i);
        }
        sort(pairs.begin(), pairs.end(), [=](std::pair<std::string, int>& a, std::pair<std::string, int>& b) {
            return a.second < b.second;
        });
        // iterate over the regions and write them out.
        for (auto const &i : pairs) {
            std::string name = i.first;
            uint64_t idx = i.second;
            region_file << idx << "\t" << name << endl;
        }
        // close the region file
        region_file.close();
        // delete the lock file, so everyone can read our data.
        std::remove(my_lock_filename.str().c_str());
    }

    void otf2_listener::write_region_map() {
        struct stat buffer;   
        std::map<std::string,uint64_t> reduced_map;
        // wait on the map file to exist
        ostringstream region_filename;
        region_filename << region_filename_prefix << 0;
        while (stat (region_filename.str().c_str(), &buffer) != 0) {}
        // wait for the lock file to not exist
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << 0;
        while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
        std::string region_line;
        std::ifstream region_file(region_filename.str());
        std::string region_name;
        int idx;
        // read the map from that rank
        while (std::getline(region_file, region_line)) {
            istringstream ss(region_line);
            ss >> idx >> region_name;
            reduced_map[region_name] = idx;
        }
        // build the array of uint64_t values
        auto region_indices = get_global_region_indices();
        uint64_t * mappings = (uint64_t*)(malloc(sizeof(uint64_t) * region_indices.size()));
        for (auto const &i : region_indices) {
            task_identifier id = i.first;
            uint64_t idx = i.second;
            uint64_t mapped_index = reduced_map[id.get_name()];
            mappings[idx] = mapped_index;
        }
        // create a map
        uint64_t map_size = region_indices.size();
        OTF2_IdMap * my_map = OTF2_IdMap_CreateFromUint64Array(map_size, mappings, false);
        for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
            OTF2_DefWriter* def_writer = getDefWriter(i);
            OTF2_DefWriter_WriteMappingTable(def_writer, OTF2_MAPPING_REGION, my_map);
            OTF2_Archive_CloseDefWriter( archive, def_writer );
        }
        OTF2_IdMap_Free(my_map);
        free(mappings);
    }

    void otf2_listener::write_clock_properties(void) {
        /* write the clock properties */
        /*
        uint64_t ticks_per_second = (uint64_t)(1.0/profiler::get_cpu_mhz());
        uint64_t globalOffset = profiler::time_point_to_nanoseconds(profiler::get_global_start());
        uint64_t traceLength = profiler::time_point_to_nanoseconds(profiler::get_global_end());
        */
        uint64_t ticks_per_second = 1000000;
        using namespace std::chrono;
        uint64_t traceLength = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        traceLength = traceLength - this->globalOffset;
        OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
            ticks_per_second /* 1,000,000,000 ticks per second */,
            //this->globalOffset /* epoch */,
            //traceLength /* length */ );
            0 /* epoch */,
            traceLength /* length */ );
        cout << my_saved_node_id << ": start: 0, stop: " << traceLength << ", ticks per second: " << ticks_per_second << endl;
    }

    void otf2_listener::write_host_properties(int rank, int pid, std::string& hostname) {
        static int max_thread_strings = 1;
        static std::map<std::string, uint64_t> hostnames;
        static const std::string node("node");
        /* define some strings */
        stringstream locality;
        locality << "process " << pid;
        OTF2_GlobalDefWriter_WriteString( global_def_writer, 
            get_string_index(locality.str()), locality.str().c_str() );
        auto tmp = hostnames.find(hostname);
        uint64_t node_index = 0;
        if (tmp == hostnames.end()) {
            node_index = hostnames.size();
            hostnames[hostname] = node_index;
            OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                get_string_index(hostname), hostname.c_str());
            OTF2_GlobalDefWriter_WriteSystemTreeNode( global_def_writer,
                node_index, /* System Tree Node ID */
                get_string_index(hostname), /* host name string ID */
                get_string_index(node), /* class name string ID */
                OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */ );
        } else {
            node_index = tmp->second;
        }
        uint64_t node_id = rank;
        node_id = node_id << 32;
        OTF2_GlobalDefWriter_WriteLocationGroup( global_def_writer,
            rank /* id */,
            get_string_index(locality.str()) /* name */,
            OTF2_LOCATION_GROUP_TYPE_PROCESS,
            node_index /* system tree node ID */ );
        /* write out the thread locations */
        for (int i = 0 ; i < rank_thread_map[rank] ; i++) {
            uint64_t thread_id = node_id + i;
            stringstream thread;
            thread << "thread " << i;
            if (i > max_thread_strings) {
                OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                    get_string_index(thread.str()), thread.str().c_str() );
                max_thread_strings++;
            }
            OTF2_GlobalDefWriter_WriteLocation( global_def_writer, 
                thread_id /* id */,
                get_string_index(thread.str()) /* name */,
                OTF2_LOCATION_TYPE_CPU_THREAD,
                rank_region_map[rank] /* number of events */,
                rank /* location group ID */ );
        }
    }
    void otf2_listener::on_shutdown(shutdown_event_data &data) {
        APEX_UNUSED(data);
        if (!_terminate) {
            _terminate = true;
            /* close event files */
            OTF2_Archive_CloseEvtFiles( archive );
            /* if we are node 0, write the global definitions */
            if (apex::__instance()->get_node_id() == 0) {
                // save my number of threads
                rank_thread_map[0] = thread_instance::get_num_threads();
                reduce_regions();
                write_region_map();
                global_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );
                OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(empty), empty.c_str() );
                write_otf2_regions();
                write_clock_properties();
                const string node("node");
                OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                    get_string_index(node), node.c_str() );
                int number_of_lines = 0;
                std::string line;
                std::ifstream myfile(index_filename);
                int rank, pid;
                std::string hostname;
                while (std::getline(myfile, line)) {
                    istringstream ss(line);
                    ss >> rank >> pid >> hostname;
                    write_host_properties(rank, pid, hostname);
                    ++number_of_lines;
                }    
            } else {
                write_my_regions();
                write_region_map();
            }
            OTF2_Archive_Close( archive );
        }
        return;
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        if (!_terminate) {
            /* We need to check in with locality/rank 0 to let
             * it know how many localities/ranks there are in
             * the job. We do that by writing our rank to the 
             * master rank file (assuming a shared filesystem)
             * if it is larger than the current rank in there. */
            pid_t pid = ::getpid();
            my_saved_node_id = apex::instance()->get_node_id();
            //if (my_saved_node_id != 0) {
                ofstream indexfile;
                char hostname[128];
                gethostname(hostname, sizeof hostname);
                string host(hostname);
                // write our pid and hostname
                indexfile.open(index_filename, ios::out | ios::ate | ios::app );
                indexfile << my_saved_node_id << "\t" << pid << "\t" << hostname << endl;
                indexfile.close();
            //}
        }
        return;
    }

    void otf2_listener::on_new_thread(new_thread_event_data &data) {
        /* the event writer and def writers are created using
         * static construction in on_start and on_stop */
        APEX_UNUSED(data);
        return;
    }

    void otf2_listener::on_exit_thread(event_data &data) {
        if (!_terminate) {
            //OTF2_Archive_CloseDefWriter( archive, getDefWriter() );
            OTF2_Archive_CloseEvtWriter( archive, getEvtWriter() );
        }
        APEX_UNUSED(data);
        return;
    }

    bool otf2_listener::on_start(task_identifier * id) {
        static __thread OTF2_EvtWriter* local_evt_writer = getEvtWriter();
        //static __thread OTF2_DefWriter* local_def_writer = getDefWriter();
        if (!_terminate) {
          /*
            profiler * p = thread_instance::instance().get_current_profiler();
            uint64_t stamp = profiler::time_point_to_nanoseconds(p->start); 
                     */
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            stamp = stamp - this->globalOffset;
            cout << my_saved_node_id << ": " << stamp << " " << id->get_name() << " start " << endl;
            OTF2_EvtWriter_Enter( local_evt_writer, NULL, 
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
        static __thread OTF2_EvtWriter* local_evt_writer = getEvtWriter();
        //static __thread OTF2_DefWriter* local_def_writer = getDefWriter();
        if (!_terminate) {
          /*
            uint64_t stamp = profiler::time_point_to_nanoseconds(p->end);
                     */
            using namespace std::chrono;
            uint64_t stamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            stamp = stamp - this->globalOffset;
            cout << my_saved_node_id << ": " << stamp << " " << p->task_id->get_name() << " stop " << endl;
            OTF2_EvtWriter_Leave( local_evt_writer, NULL, 
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
