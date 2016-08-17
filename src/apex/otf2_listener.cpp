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
#include <fcntl.h>
#include <sys/file.h>

using namespace std;

namespace apex {

    uint64_t otf2_listener::globalOffset(0);
    const std::string otf2_listener::empty("");
    __thread OTF2_EvtWriter* otf2_listener::evt_writer(nullptr);
    OTF2_EvtWriter* otf2_listener::comm_evt_writer(nullptr);
    const std::string otf2_listener::index_filename("./.max_locality.txt");
    const std::string otf2_listener::region_filename_prefix("./.regions.");
    const std::string otf2_listener::lock_filename_prefix("./.regions.lock.");
    int otf2_listener::my_saved_node_id(0);

    OTF2_CallbackCode otf2_listener::my_OTF2GetSize(void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *size) {
        /* Returns the number of OTF2_Archive objects operating in this
           communication context. */
        //cout << __func__ << endl;
        *size = apex_options::otf2_collective_size();
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2GetRank (void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *rank) {
        /* Returns the rank of this OTF2_Archive objects in this communication
           context. A number between 0 and one less of the size of the communication
           context. */
        //cout << __func__ << endl;
        *rank = my_saved_node_id;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2CreateLocalComm (void *userData,
            OTF2_CollectiveContext **localCommContext, OTF2_CollectiveContext
            *globalCommContext, uint32_t globalRank, uint32_t globalSize, uint32_t
            localRank, uint32_t localSize, uint32_t fileNumber, uint32_t numberOfFiles) {
        /* Create a new disjoint partitioning of the the globalCommContext
           communication context. numberOfFiles denotes the number of the partitions.
           fileNumber denotes in which of the partitions this OTF2_Archive should belong.
           localSize is the size of this partition and localRank the rank of this
           OTF2_Archive in the partition. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2FreeLocalComm (void *userData,
            OTF2_CollectiveContext *localCommContext) {
        /* Destroys the communication context previous created by the
           OTF2CreateLocalComm callback. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Barrier (void *userData,
            OTF2_CollectiveContext *commContext) {
        /* Performs a barrier collective on the given communication context. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Bcast (void *userData,
            OTF2_CollectiveContext *commContext, void *data, uint32_t numberElements,
            OTF2_Type type, uint32_t root) {
        /* Performs a broadcast collective on the given communication context. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gather (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute the same number of elements. outData is only valid at
           rank root. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gatherv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, uint32_t inElements,
            void *outData, const uint32_t *outElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute different number of elements. outData and outElements are
           only valid at rank root. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatter (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute the same number of elements. inData is only valid at rank
           root. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatterv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, const uint32_t
            *inElements, void *outData, uint32_t outElements, OTF2_Type type, uint32_t
            root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute different number of elements. inData and inElements are
           only valid at rank root. */
        //cout << __func__ << endl;
        return OTF2_CALLBACK_SUCCESS;
    }

    void otf2_listener::my_OTF2Release (void *userData, OTF2_CollectiveContext
            *globalCommContext, OTF2_CollectiveContext *localCommContext) {
        /* Optionally called in OTF2_Archive_Close or OTF2_Reader_Close
           respectively. */
        //cout << __func__ << endl;
        return;
    }

    OTF2_CollectiveCallbacks * otf2_listener::get_collective_callbacks (void) {
        static OTF2_CollectiveCallbacks cb;
        cb.otf2_release = my_OTF2Release;
        cb.otf2_get_size = my_OTF2GetSize;
        cb.otf2_get_rank = my_OTF2GetRank;
        cb.otf2_create_local_comm = my_OTF2CreateLocalComm;
        cb.otf2_free_local_comm = my_OTF2FreeLocalComm;
        cb.otf2_barrier = my_OTF2Barrier;
        cb.otf2_bcast = my_OTF2Bcast;
        cb.otf2_gather = my_OTF2Gather;
        cb.otf2_gatherv = my_OTF2Gatherv;
        cb.otf2_scatter = my_OTF2Scatter;
        cb.otf2_scatterv = my_OTF2Scatterv;
        return &cb;
    }

    OTF2_EvtWriter* otf2_listener::getEvtWriter(void) {
      if (evt_writer == nullptr) {
        uint64_t my_node_id = apex::__instance()->get_node_id();
        my_node_id = (my_node_id << 32) + thread_instance::get_id();
        evt_writer = OTF2_Archive_GetEvtWriter( archive, my_node_id );
        if (thread_instance::get_id() == 0) {
            comm_evt_writer = evt_writer;
        }
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

    otf2_listener::otf2_listener (void) : _terminate(false), global_def_writer(nullptr) {
        /* get a start time for the trace */
        globalOffset = get_time();
        /* set the flusher */
        flush_callbacks = { 
            .otf2_pre_flush  = otf2_listener::pre_flush, 
            .otf2_post_flush = otf2_listener::post_flush 
        };
    }

    bool otf2_listener::create_archive(void) {
        /* only open once! */
        static bool created = false;
        if (created) return true;

        /* open the OTF2 archive */
        archive = OTF2_Archive_Open( apex_options::otf2_archive_path(),
                apex_options::otf2_archive_name(),
                OTF2_FILEMODE_WRITE,
                OTF2_CHUNK_SIZE_EVENTS_DEFAULT,
                OTF2_CHUNK_SIZE_DEFINITIONS_DEFAULT,
                OTF2_SUBSTRATE_POSIX,
                OTF2_COMPRESSION_NONE );
        /* set the flush callbacks, basically getting timestamps */
        OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks, NULL );
        /* set the creator name */
        stringstream tmp;
        tmp << "APEX version " << version();
        OTF2_Archive_SetCreator(archive, tmp.str().c_str());
        /* we have no collective callbacks. */
        if (OTF2_Archive_SetCollectiveCallbacks(archive, get_collective_callbacks(), NULL, NULL, NULL) != OTF2_SUCCESS)
        /* open the event files for this archive */
        OTF2_Archive_OpenEvtFiles( archive );
        created = true;
        return created;
     }

    void otf2_listener::on_startup(startup_event_data &data) {
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
            //uint64_t idx = i.second;
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
        uint64_t ticks_per_second = 1e9;
        uint64_t traceLength = get_time();
        OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
            ticks_per_second, 0 /* start */, traceLength /* length */ );
    }

    void otf2_listener::write_host_properties(int rank, int pid, std::string& hostname) {
        static std::set<std::string> threadnames;
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
            auto tmp = threadnames.find(thread.str());
            if (tmp == threadnames.end()) {
                OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                    get_string_index(thread.str()), thread.str().c_str() );
                threadnames.insert(thread.str());
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
            if (my_saved_node_id == 0) {
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
                std::string line;
                std::ifstream myfile(index_filename);
                int rank, pid;
                std::string hostname;
                std::map<int,int> rank_pid_map;
                std::map<int,string> rank_hostname_map;
                while (std::getline(myfile, line)) {
                    istringstream ss(line);
                    ss >> rank >> pid >> hostname;
                    rank_pid_map[rank] = pid;
                    rank_hostname_map[rank] = hostname;
                }    
                /* should we have a location for all threads, or just the master threads? */
                vector<uint64_t>group_members;
                vector<uint64_t>group_members_t0;
                for (auto const &i : rank_pid_map) {
                    rank = i.first;
                    pid = i.second;
                    hostname = rank_hostname_map[rank];
                    write_host_properties(rank, pid, hostname);
                    group_members.push_back(rank);
                    group_members_t0.push_back(group_members[rank] << 32);
                }
                const char * world = "MPI_COMM_WORLD";
                const char * world_group = "MPI_COMM_WORLD_GROUP";
                const char * world_locations = "MPI_COMM_WORLD_LOCATIONS";
                OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world), world );
                OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world_locations), world_locations );
                OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world_group), world_group );
                OTF2_GlobalDefWriter_WriteGroup ( global_def_writer,
                    0, get_string_index(world_locations), OTF2_GROUP_TYPE_COMM_LOCATIONS,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE, group_members_t0.size(),
                    &group_members_t0[0]);   
                OTF2_GlobalDefWriter_WriteGroup ( global_def_writer,
                    0, get_string_index(world_group), OTF2_GROUP_TYPE_COMM_GROUP,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE, group_members.size(),
                    &group_members[0]);   
                OTF2_GlobalDefWriter_WriteComm  ( global_def_writer,
                    0, get_string_index(world), 
                    0, OTF2_UNDEFINED_COMM);
            } else {
                write_my_regions();
                write_region_map();
            }
            OTF2_Archive_Close( archive );
        }
        return;
    }
    
    bool otf2_listener::write_my_node_properties() {
        static bool already_written = false;
        if (already_written) return true;
        /* We need to check in with locality/rank 0 to let
         * it know how many localities/ranks there are in
         * the job. We do that by writing our rank to the 
         * master rank file (assuming a shared filesystem)
         * if it is larger than the current rank in there. */
        pid_t pid = ::getpid();
        char hostname[128];
        gethostname(hostname, sizeof hostname);
        string host(hostname);
        stringstream ss;
        ss << my_saved_node_id << "\t" << pid << "\t" << hostname << "\n";
        string tmp = ss.str();
        // write our pid and hostname, using low-level file locking!
        struct flock fl;
        fl.l_type   = F_WRLCK;  /* F_RDLCK, F_WRLCK, F_UNLCK    */
        fl.l_whence = SEEK_SET; /* SEEK_SET, SEEK_CUR, SEEK_END */
        fl.l_start  = 0;        /* Offset from l_whence         */
        fl.l_len    = 0;        /* length, 0 = to EOF           */
        fl.l_pid    = pid;      /* our PID                      */
        int indexfile = open(index_filename.c_str(), O_APPEND | O_WRONLY );
        assert(indexfile >= 0);
        fcntl(indexfile, F_SETLKW, &fl);  /* F_GETLK, F_SETLK, F_SETLKW */
        assert(write(indexfile, tmp.c_str(), tmp.size()) >= 0);
        fl.l_type   = F_UNLCK;   /* tell it to unlock the region */
        fcntl(indexfile, F_SETLK, &fl); /* set the region to unlocked */
        assert(close(indexfile) >= 0);
        already_written = true;
        return already_written;
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        my_saved_node_id = apex::instance()->get_node_id();
        static bool archive_created = create_archive();
        if ((!_terminate) && archive_created) {
            write_my_node_properties();
            // set up the event writer for communication
            getEvtWriter();
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
        // before we process the event, make sure the archive is open
        // THIS WILL ONLY HAPPEN ONCE
        static bool archive_created = create_archive();
        // before we process the event, make sure the node properties are written
        // THIS WILL ONLY HAPPEN ONCE
        static bool properties_written = write_my_node_properties();
        // before we process the event, make sure the event write is open
        // THIS WILL ONLY HAPPEN ONCE
        static __thread OTF2_EvtWriter* local_evt_writer = getEvtWriter();
        if ((!_terminate) && archive_created && properties_written) {
            /*
            using namespace std::chrono;
            profiler * p = thread_instance::instance().get_current_profiler();
            uint64_t stamp = p->start.time_since_epoch().count() - globalOffset;
            */
            if (thread_instance::get_id() == 0) {
                std::unique_lock<std::mutex> lock(_mutex);
                uint64_t stamp = get_time();
                OTF2_EvtWriter_Enter( local_evt_writer, NULL, stamp, get_region_index(id) /* region */ );
            } else {
                uint64_t stamp = get_time();
                OTF2_EvtWriter_Enter( local_evt_writer, NULL, stamp, get_region_index(id) /* region */ );
            }
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
        if (!_terminate) {
          /*
            using namespace std::chrono;
            uint64_t stamp = p->end.time_since_epoch().count() - globalOffset;
                     */
            if (thread_instance::get_id() == 0) {
                std::unique_lock<std::mutex> lock(_mutex);
                uint64_t stamp = get_time();
                OTF2_EvtWriter_Leave( local_evt_writer, NULL, stamp, 
                        get_region_index(p->task_id) /* region */ );
            } else {
                uint64_t stamp = get_time();
                OTF2_EvtWriter_Leave( local_evt_writer, NULL, stamp, 
                        get_region_index(p->task_id) /* region */ );
            }
        }
        return;
    }

    void otf2_listener::on_yield(std::shared_ptr<profiler> &p) {
        on_stop(p);
    }

    void otf2_listener::on_send(message_event_data &data) {
        if (!_terminate && comm_evt_writer != NULL) {
            OTF2_AttributeList * attributeList = OTF2_AttributeList_New();
            OTF2_CommRef communicator = 0;
            {
                std::unique_lock<std::mutex> lock(_comm_mutex);
                uint64_t stamp = get_time();
                OTF2_EvtWriter_MpiSend  ( comm_evt_writer,
                        attributeList, stamp, data.target, communicator,
                        data.id, data.size );
            }
            OTF2_AttributeList_Delete(attributeList);
        }
        return;
    }

    void otf2_listener::on_recv(message_event_data &data) {
        if (!_terminate && comm_evt_writer != NULL) {
            OTF2_AttributeList * attributeList = OTF2_AttributeList_New();
            OTF2_CommRef communicator = 0;
            {
                std::unique_lock<std::mutex> lock(_comm_mutex);
                uint64_t stamp = get_time();
                OTF2_EvtWriter_MpiRecv  ( comm_evt_writer,
                        attributeList, stamp, data.source, communicator,
                        data.id, data.size );
            }
            OTF2_AttributeList_Delete(attributeList);
        }
        return;
    }

}
