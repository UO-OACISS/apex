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

#define OTF2_EC(call) { \
    OTF2_ErrorCode ec = call; \
    if (ec != OTF2_SUCCESS) { \
        printf("OTF2 Error: %s, %s\n", OTF2_Error_GetName(ec), OTF2_Error_GetDescription (ec)); \
    } \
}

using namespace std;

namespace apex {

    uint64_t otf2_listener::globalOffset(0);
    const std::string otf2_listener::empty("");
    int otf2_listener::my_saved_node_id(0);
    int otf2_listener::my_saved_node_count(1);

    OTF2_CallbackCode otf2_listener::my_OTF2GetSize(void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *size) {
        /* Returns the number of OTF2_Archive objects operating in this
           communication context. */
        //cout << __func__ << " " << apex_options::otf2_collective_size() << endl;
        //*size = apex_options::otf2_collective_size();
        *size = my_saved_node_count;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2GetRank (void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *rank) {
        /* Returns the rank of this OTF2_Archive objects in this communication
           context. A number between 0 and one less of the size of the communication
           context. */
        //cout << __func__ << " " << my_saved_node_id << endl;
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
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2FreeLocalComm (void *userData,
            OTF2_CollectiveContext *localCommContext) {
        /* Destroys the communication context previous created by the
           OTF2CreateLocalComm callback. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Barrier (void *userData,
            OTF2_CollectiveContext *commContext) {
        /* Performs a barrier collective on the given communication context. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Bcast (void *userData,
            OTF2_CollectiveContext *commContext, void *data, uint32_t numberElements,
            OTF2_Type type, uint32_t root) {
        /* Performs a broadcast collective on the given communication context. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gather (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute the same number of elements. outData is only valid at
           rank root. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gatherv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, uint32_t inElements,
            void *outData, const uint32_t *outElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute different number of elements. outData and outElements are
           only valid at rank root. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatter (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute the same number of elements. inData is only valid at rank
           root. */
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatterv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, const uint32_t
            *inElements, void *outData, uint32_t outElements, OTF2_Type type, uint32_t
            root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute different number of elements. inData and inElements are
           only valid at rank root. */
        return OTF2_CALLBACK_SUCCESS;
    }

    void otf2_listener::my_OTF2Release (void *userData, OTF2_CollectiveContext
            *globalCommContext, OTF2_CollectiveContext *localCommContext) {
        /* Optionally called in OTF2_Archive_Close or OTF2_Reader_Close
           respectively. */
        return;
    }

        /* these indices are thread-specific. */
        std::map<task_identifier,uint64_t>& otf2_listener::get_region_indices(void) {
            static __thread std::map<task_identifier,uint64_t> * region_indices;
            if (region_indices == nullptr) {
                region_indices = new std::map<task_identifier,uint64_t>();
            }
            return *region_indices;
        }
        /* these indices are thread-specific. */
        std::map<std::string,uint64_t>& otf2_listener::get_string_indices(void) {
            static __thread std::map<std::string,uint64_t> * string_indices;
            if (string_indices == nullptr) {
                string_indices = new std::map<std::string,uint64_t>();
            }
            return *string_indices;
        }
        uint64_t otf2_listener::get_region_index(task_identifier* id) {
            /* first, look in this thread's map */
            std::map<task_identifier,uint64_t>& region_indices = get_region_indices();
            auto tmp = region_indices.find(*id);
            uint64_t region_index = 0;
            if (tmp == region_indices.end()) {
                /* not in the thread's map? look in the global map */
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
        uint64_t otf2_listener::get_string_index(const std::string& name) {
            // thread specific
              std::map<std::string,uint64_t>& string_indices = get_string_indices();
            // process specific
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
        uint64_t otf2_listener::get_hostname_index(const std::string& name) {
            /* first, look in the map */
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
        std::map<std::string,uint64_t>& otf2_listener::get_metric_indices(void) {
            static __thread std::map<std::string,uint64_t> * metric_indices;
            if (metric_indices == nullptr) {
                metric_indices = new std::map<std::string,uint64_t>();
            }
            return *metric_indices;
        }
        uint64_t otf2_listener::get_metric_index(const std::string& name) {
            // thread specific
              std::map<std::string,uint64_t>& metric_indices = get_metric_indices();
            /* first, look in this thread's map */
            auto tmp = metric_indices.find(name);
            uint64_t metric_index = 0;
            if (tmp == metric_indices.end()) {
                // process specific
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
      static __thread OTF2_EvtWriter* evt_writer(nullptr);
      if (evt_writer == nullptr) {
        // only let one thread at a time create an event file
        read_lock_type lock(_archive_mutex);
        //printf("creating event writer for thread %lu\n", thread_instance::get_id()); fflush(stdout);
        uint64_t my_node_id = my_saved_node_id;
        my_node_id = (my_node_id << 32) + thread_instance::get_id();
        evt_writer = OTF2_Archive_GetEvtWriter( archive, my_node_id );
        if (thread_instance::get_id() == 0) {
            comm_evt_writer = evt_writer;
        }
        std::unique_lock<std::mutex> l(_event_set_mutex);
        _event_threads.insert(thread_instance::get_id());
      }
      return evt_writer;
    }

    bool otf2_listener::event_file_exists (int threadid) {
        // get exclusive access to the set - unlocks on exit
        std::unique_lock<std::mutex> l(_event_set_mutex);
        if (_event_threads.find(threadid) == _event_threads.end()) {return false;} else {return true;}
    }

    OTF2_DefWriter* otf2_listener::getDefWriter(int threadid) {
        OTF2_DefWriter* def_writer;
        //printf("creating definition writer for thread %d\n", threadid); fflush(stdout);
        uint64_t my_node_id = my_saved_node_id;
        my_node_id = (my_node_id << 32) + threadid;
        def_writer = OTF2_Archive_GetDefWriter( archive, my_node_id );
        return def_writer;
    }

    OTF2_FlushCallbacks otf2_listener::flush_callbacks =
    {
        .otf2_pre_flush  = pre_flush,
        .otf2_post_flush = post_flush
    };

    /* constructor for the OTF2 listener class */
    otf2_listener::otf2_listener (void) : _terminate(false), comm_evt_writer(nullptr), global_def_writer(nullptr) {
        /* get a start time for the trace */
        globalOffset = get_time();
        /* set the flusher */
        flush_callbacks = { 
            .otf2_pre_flush  = otf2_listener::pre_flush, 
            .otf2_post_flush = otf2_listener::post_flush 
        };
        index_filename = string(string(apex_options::otf2_archive_path()) + "/.locality.");
        region_filename_prefix = string(string(apex_options::otf2_archive_path()) + "/.regions.");
        metric_filename_prefix = string(string(apex_options::otf2_archive_path()) + "/.metrics.");
        lock_filename_prefix = string(string(apex_options::otf2_archive_path()) + "/.regions.lock.");
    }

    bool otf2_listener::create_archive(void) {
        /* only one thread per process allowed in here, ever! */
        write_lock_type lock(_archive_mutex);
        /* only open once! */
        static bool created = false;
        if (created) return true;

        if (apex_options::otf2_testing() && my_saved_node_id == 0) {
            // is this a good idea?
            /* NO! why? because we don't know which rank we are, and
             * we don't know if the archive is supposed to be there or not.
             */
            remove_path(apex_options::otf2_archive_path());
        }
        /* open the OTF2 archive */
        archive = OTF2_Archive_Open( apex_options::otf2_archive_path(),
                apex_options::otf2_archive_name(),
                OTF2_FILEMODE_WRITE,
                OTF2_CHUNK_SIZE_EVENTS_DEFAULT,
                OTF2_CHUNK_SIZE_DEFINITIONS_DEFAULT,
                OTF2_SUBSTRATE_POSIX,
                OTF2_COMPRESSION_NONE );
        /* set the flush callbacks, basically getting timestamps */
        OTF2_EC(OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks, NULL ));
        /* set the creator name */
        stringstream tmp;
        tmp << "APEX version " << version();
        OTF2_EC(OTF2_Archive_SetCreator(archive, tmp.str().c_str()));
        /* we have no collective callbacks. */
        OTF2_EC(OTF2_Archive_SetCollectiveCallbacks(archive, 
            get_collective_callbacks(), NULL, NULL, NULL));
        /* open the event files for this archive */
        OTF2_EC(OTF2_Archive_OpenEvtFiles( archive ));
        created = true;
        return created;
     }

    void otf2_listener::on_startup(startup_event_data &data) {
       // add the empty string to the string definitions
        get_string_index(empty);
        // save the node id, because the apex object my not be
        // around when we are finalizing everything.
        my_saved_node_id = apex::instance()->get_node_id();
        my_saved_node_count = apex::instance()->get_num_ranks();
        // now is a good time to make sure the archive is open on this rank/locality
        static bool archive_created = create_archive();
        if ((!_terminate) && archive_created) {
            // let rank/locality 0 know this rank's properties.
            write_my_node_properties();
            // set up the event writer for communication (thread 0).
            getEvtWriter();
        }
        return;
    }

    void otf2_listener::write_otf2_regions(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        for (auto const &i : reduced_region_map) {
            string id = i.first;
            uint64_t idx = i.second;
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(id), id.c_str() );
			size_t found = id.find(string("UNRESOLVED"));
            OTF2_GlobalDefWriter_WriteRegion( global_def_writer,
                    idx /* id */,
                    get_string_index(id) /* region name  */,
                    get_string_index(empty) /* alternative name */,
                    get_string_index(empty) /* description */,
                    (found != std::string::npos) ? OTF2_REGION_ROLE_ARTIFICIAL : OTF2_REGION_ROLE_TASK,
					(found != std::string::npos) ? OTF2_PARADIGM_MEASUREMENT_SYSTEM : OTF2_PARADIGM_USER,
                    OTF2_REGION_FLAG_NONE,
                    get_string_index(empty) /* source file */,
                    get_string_index(empty) /* begin lno */,
                    get_string_index(empty) /* end lno */ );
        }
    }

    void otf2_listener::write_otf2_metrics(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        // write a "unit" string
        OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index("count"), "count" );
        // copy the reduced map to a pair, so we can sort by value
        std::vector<std::pair<std::string, int>> pairs;
        for (auto const &i : reduced_metric_map) {
            pairs.push_back(i);
        }
        sort(pairs.begin(), pairs.end(), [=](std::pair<std::string, int>& a, std::pair<std::string, int>& b) {
            return a.second < b.second;
        });
        // iterate over the metrics and write them out.
        for (auto const &i : pairs) {
            string id = i.first;
            uint64_t idx = i.second;
            OTF2_GlobalDefWriter_WriteString( global_def_writer, get_string_index(id), id.c_str() );
              OTF2_GlobalDefWriter_WriteMetricMember( global_def_writer,
                idx, get_string_index(id), get_string_index(id),
                OTF2_METRIC_TYPE_OTHER, OTF2_METRIC_ABSOLUTE_POINT, 
                OTF2_TYPE_DOUBLE, OTF2_BASE_DECIMAL, 0, get_string_index("count"));
            OTF2_MetricMemberRef omr[1];
            omr[0]=idx;
            OTF2_GlobalDefWriter_WriteMetricClass( global_def_writer, 
                    idx, 1, omr, OTF2_METRIC_ASYNCHRONOUS, 
                    OTF2_RECORDER_KIND_UNKNOWN);
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
        for (auto const &i : global_region_indices) {
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

    void otf2_listener::write_my_metrics(void) {
        // only write these out once!
        static __thread bool written = false;
        if (written) return;
        written = true;
        // create my lock file.
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // open my metric file
        ostringstream metric_filename;
        metric_filename << metric_filename_prefix << my_saved_node_id;
        ofstream metric_file(metric_filename.str(), ios::out | ios::trunc );
        // first, output our number of threads.
        metric_file << thread_instance::get_num_threads() << endl;
        // then iterate over the metrics and write them out.
        for (auto const &i : global_metric_indices) {
            string id = i.first;
            //uint64_t idx = i.second;
            //metric_file << id.get_name() << "\t" << idx << endl;
            metric_file << id << endl;
        }
        // close the metric file
        metric_file.close();
        // delete the lock file, so rank 0 can read our data.
        std::remove(lock_filename.str().c_str());
    }

    int otf2_listener::reduce_regions(void) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(my_lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // iterate over my region map, and build a map of strings to ids
        // save my number of regions
        rank_region_map[0] = global_region_indices.size();
        for (auto const &i : global_region_indices) {
            task_identifier id = i.first;
            uint64_t idx = i.second;
            reduced_region_map[id.get_name()] = idx;
        }
        int comm_size = 0;
        // iterate over the other ranks in the index files
        for (int i = 0 ; i < my_saved_node_count ; i++) {
            comm_size++;
            // skip myself
            if (i == 0) continue;
            rank_region_map[i] = 0;
            struct stat buffer;   
            // wait on the map file to exist
            ostringstream region_filename;
            region_filename << region_filename_prefix << i;
            // wait for the lock file to not exist
            while (stat (region_filename.str().c_str(), &buffer) != 0) {}
            ostringstream lock_filename;
            lock_filename << lock_filename_prefix << i;
            // wait for the region file to exist
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            // get the number of threads from that rank
            std::string region_line;
            std::ifstream region_file(region_filename.str());
            std::getline(region_file, region_line);
            std::string::size_type sz;   // alias of size_t
            rank_thread_map[i] = std::stoi(region_line,&sz);
            // read the map from that rank
            while (std::getline(region_file, region_line)) {
                rank_region_map[i] = rank_region_map[i] + 1;
                // trim the newline
                region_line.erase(std::remove(region_line.begin(), region_line.end(), '\n'), region_line.end());
                if (reduced_region_map.find(region_line) == reduced_region_map.end()) {
                    uint64_t idx = reduced_region_map.size();
                    reduced_region_map[region_line] = idx;
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
        for (auto const &i : reduced_region_map) {
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
        return comm_size;
    }

    void otf2_listener::reduce_metrics(void) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(my_lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // iterate over my metric map, and build a map of strings to ids
        // save my number of metrics
        rank_metric_map[0] = global_metric_indices.size();
        for (auto const &i : global_metric_indices) {
            string id = i.first;
            uint64_t idx = i.second;
            reduced_metric_map[id] = idx;
        }
        // iterate over the other ranks in the index file
        for (int i = 0 ; i < my_saved_node_count ; i++) {
            // skip myself
            if (i == 0) continue;
            rank_metric_map[i] = 0;
            struct stat buffer;   
            // wait on the map file to exist
            ostringstream metric_filename;
            metric_filename << metric_filename_prefix << i;
            while (stat (metric_filename.str().c_str(), &buffer) != 0) {}
            // wait for the lock file to not exist
            ostringstream lock_filename;
            lock_filename << lock_filename_prefix << i;
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            // get the number of threads from that rank
            std::string metric_line;
            std::ifstream metric_file(metric_filename.str());
            std::getline(metric_file, metric_line);
            std::string::size_type sz;   // alias of size_t
            rank_thread_map[i] = std::stoi(metric_line,&sz);
            // read the map from that rank
            while (std::getline(metric_file, metric_line)) {
                rank_metric_map[i] = rank_metric_map[i] + 1;
                // trim the newline
                metric_line.erase(std::remove(metric_line.begin(), metric_line.end(), '\n'), metric_line.end());
                if (reduced_metric_map.find(metric_line) == reduced_metric_map.end()) {
                    uint64_t idx = reduced_metric_map.size();
                    reduced_metric_map[metric_line] = idx;
                }
            }
            // close the metric file
            metric_file.close();
            // remove that rank's map
            std::remove(metric_filename.str().c_str());
        }
        // open my metric file
        ostringstream metric_filename;
        metric_filename << metric_filename_prefix << my_saved_node_id;
        ofstream metric_file(metric_filename.str(), ios::out | ios::trunc );
        // copy the reduced map to a pair, so we can sort by value
        std::vector<std::pair<std::string, int>> pairs;
        for (auto const &i : reduced_metric_map) {
            pairs.push_back(i);
        }
        sort(pairs.begin(), pairs.end(), [=](std::pair<std::string, int>& a, std::pair<std::string, int>& b) {
            return a.second < b.second;
        });
        // iterate over the metrics and write them out.
        for (auto const &i : pairs) {
            std::string name = i.first;
            uint64_t idx = i.second;
            metric_file << idx << "\t" << name << endl;
        }
        // close the metric file
        metric_file.close();
        // delete the lock file, so everyone can read our data.
        std::remove(my_lock_filename.str().c_str());
    }

    void otf2_listener::write_region_map() {
        struct stat buffer;   
        std::map<std::string,uint64_t> reduced_region_map;
        // wait on the map file from rank 0 to exist
        ostringstream region_filename;
        region_filename << region_filename_prefix << 0;
        while (stat (region_filename.str().c_str(), &buffer) != 0) {}
        // wait for the lock file from rank 0 to NOT exist
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << 0;
        while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
        std::string region_line;
        std::ifstream region_file(region_filename.str());
        std::string region_name;
        int idx;
        // read the map from rank 0
        while (std::getline(region_file, region_line)) {
            istringstream ss(region_line);
            ss >> idx >> region_name;
            reduced_region_map[region_name] = idx;
        }
        region_file.close();
        // build the array of uint64_t values
        if (global_region_indices.size() > 0) {
            uint64_t * mappings = (uint64_t*)(malloc(sizeof(uint64_t) * global_region_indices.size()));
            for (auto const &i : global_region_indices) {
                task_identifier id = i.first;
                uint64_t idx = i.second;
                uint64_t mapped_index = reduced_region_map[id.get_name()];
                mappings[idx] = mapped_index;
            }
            // create a map
            uint64_t map_size = global_region_indices.size();
            OTF2_IdMap * my_map = OTF2_IdMap_CreateFromUint64Array(map_size, mappings, false);
            for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
                if (event_file_exists(i)) {
                    OTF2_DefWriter_WriteMappingTable(getDefWriter(i), OTF2_MAPPING_REGION, my_map);
                }
            }
            // free the map
            OTF2_IdMap_Free(my_map);
            free(mappings);
        }
    }

    void otf2_listener::write_metric_map() {
        struct stat buffer;   
        std::map<std::string,uint64_t> reduced_metric_map;
        // wait on the map file from rank 0 to exist
        ostringstream metric_filename;
        metric_filename << metric_filename_prefix << 0;
        while (stat (metric_filename.str().c_str(), &buffer) != 0) {}
        // wait for the lock file from rank 0 to NOT exist
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << 0;
        while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
        std::string metric_line;
        std::ifstream metric_file(metric_filename.str());
        std::string metric_name;
        uint64_t idx;
        // read the map from rank 0
        while (std::getline(metric_file, metric_line)) {
            size_t firsttab=metric_line.find('\t');
            idx = atoi(metric_line.substr(0,firsttab).c_str());
            metric_name = metric_line.substr(firsttab+1);
            reduced_metric_map[metric_name] = idx;
        }
        metric_file.close();
        // build the array of uint64_t values
        if (global_metric_indices.size() > 0) {
            uint64_t * mappings = (uint64_t*)(malloc(sizeof(uint64_t) * global_metric_indices.size()));
            for (auto const &i : global_metric_indices) {
                string name = i.first;
                uint64_t idx = i.second;
                uint64_t mapped_index = reduced_metric_map[name];
                mappings[idx] = mapped_index;
            }
            // create a map
            uint64_t map_size = global_metric_indices.size();
            OTF2_IdMap * my_map = OTF2_IdMap_CreateFromUint64Array(map_size, mappings, false);
            for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
                if (event_file_exists(i)) {
                    OTF2_DefWriter_WriteMappingTable(getDefWriter(i), OTF2_MAPPING_METRIC, my_map);
                }
            }
            // free the map
            OTF2_IdMap_Free(my_map);
            free(mappings);
        }
    }

    void otf2_listener::write_clock_properties(void) {
        /* write the clock properties */
        uint64_t ticks_per_second = 1e9;
        uint64_t traceLength = get_time();
        OTF2_GlobalDefWriter_WriteClockProperties( global_def_writer,
            ticks_per_second, 0 /* start */, traceLength /* length */ );
    }

    /* For this rank, pid, hostname, write all that data into the
     * trace definition */
    void otf2_listener::write_host_properties(int rank, int pid, std::string& hostname) {
        static std::set<std::string> threadnames;
        static std::map<std::string, uint64_t> hostnames;
        static const std::string node("node");
        // have we written this host name before?
        auto tmp = hostnames.find(hostname);
        uint64_t node_index = 0;
        // if not, write it out
        if (tmp == hostnames.end()) {
            node_index = hostnames.size();
            hostnames[hostname] = node_index;
            // write the hostname string
            OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                get_string_index(hostname), hostname.c_str());
            // add our host to the system tree
            OTF2_GlobalDefWriter_WriteSystemTreeNode( global_def_writer,
                node_index, /* System Tree Node ID */
                get_string_index(hostname), /* host name string ID */
                get_string_index(node), /* class name string ID */
                OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */ );
        } else {
            node_index = tmp->second;
        }
        // map our rank to a globally unique ID.
        // we don't know how many threads there are for each
        // rank at startup, so each rank location is bit shifted.
        uint64_t node_id = rank;
        node_id = node_id << 32;
        // write out our process id!
        stringstream locality;
        locality << "process " << pid;
        // write our process name to the trace
        OTF2_GlobalDefWriter_WriteString( global_def_writer, 
            get_string_index(locality.str()), locality.str().c_str() );
        // write the process location to the system tree
        OTF2_GlobalDefWriter_WriteLocationGroup( global_def_writer,
            rank /* id */,
            get_string_index(locality.str()) /* name */,
            OTF2_LOCATION_GROUP_TYPE_PROCESS,
            node_index /* system tree node ID */ );
        // write out the thread locations
        for (int i = 0 ; i < rank_thread_map[rank] ; i++) {
            uint64_t thread_id = node_id + i;
            stringstream thread;
            thread << "thread " << i;
            // have we written this thread name before?
            auto tmp = threadnames.find(thread.str());
            if (tmp == threadnames.end()) {
                OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                    get_string_index(thread.str()), thread.str().c_str() );
                threadnames.insert(thread.str());
            }
            // write out the thread location into the system tree
            OTF2_GlobalDefWriter_WriteLocation( global_def_writer, 
                thread_id /* id */,
                get_string_index(thread.str()) /* name */,
                OTF2_LOCATION_TYPE_CPU_THREAD,
                rank_region_map[rank] /* number of events */,
                rank /* location group ID */ );
        }
    }

    /* At shutdown, we need to reduce all the global information,
     * and write out the global definitions - strings, regions,
     * locations, communicators, groups, metrics, etc.
     */
    void otf2_listener::on_shutdown(shutdown_event_data &data) {
        static bool _finalized = false;
        if (_finalized) { return; }
        _finalized = true;
         // get an exclusive lock, to make sure no other threads
        // are writing to the archive.
        write_lock_type lock(_archive_mutex);
        APEX_UNUSED(data);
        if (!_terminate) {
            _terminate = true;
            /* sleep a tiny bit, to make sure all other threads get the word
             * that we are done. */
			std::cout << "Waiting for all support threads to exit..." << std::endl;
            usleep(apex_options::policy_drain_timeout()); // sleep 1ms (default)
            /* close event files */
			std::cout << "Closing OTF2 event files..." << std::endl;
            OTF2_EC(OTF2_Archive_CloseEvtFiles( archive ));
            /* if we are node 0, write the global definitions */
            if (my_saved_node_id == 0) {
                // save my number of threads
                rank_thread_map[0] = thread_instance::get_num_threads();
				std::cout << "Writing OTF2 definition files..." << std::endl;
                // make a common list of regions and metrics across all nodes...
                reduce_regions();
                reduce_metrics();
                if (my_saved_node_count > 1) {
                    // ...and distribute them back out
                    write_region_map();
                    write_metric_map();
                }
				std::cout << "Writing OTF2 Global definition file..." << std::endl;
                // create the global definition writer
                global_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );
                // write an "empty" string - only once
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(empty), empty.c_str() ));
                // write out the reduced set of regions
                write_otf2_regions();
                // write out the reduced set of metrics
                write_otf2_metrics();
                // write out the clock properties
                write_clock_properties();
                // write a "node" string - only once
                const string node("node");
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer, 
                    get_string_index(node), node.c_str() ));
                std::map<int,int> rank_pid_map;
                std::map<int,string> rank_hostname_map;
                int rank, pid;
                std::string hostname;
                // iterate over the node info file, getting
                // the rank, pid and hostname for each
                for (int i = 0 ; i < my_saved_node_count ; i++) {
                    std::string line;
        			struct stat buffer;
                    std::stringstream full_index_filename;
                    full_index_filename << index_filename << to_string(i);
					// wait for the file to exist
        			while (stat (full_index_filename.str().c_str(), &buffer) != 0) {}
        			std::ifstream myfile(full_index_filename.str());
                    while (std::getline(myfile, line)) {
                        istringstream ss(line);
                        ss >> rank >> pid >> hostname;
                        rank_pid_map[rank] = pid;
                        rank_hostname_map[rank] = hostname;
                    }    
                    myfile.close();
                }
                // these are communicator lists, and a location map
                // for each. We need a group member for each process,
                // and the "location" is thread 0 of that process.
                vector<uint64_t>group_members;
                vector<uint64_t>group_members_threads;
				std::cout << "Writing OTF2 Node information..." << std::endl;
                // iterate over the ranks (in order) and write them out
                for (auto const &i : rank_pid_map) {
                    rank = i.first;
                    pid = i.second;
                    hostname = rank_hostname_map[rank];
                    // write the host properties to the OTF2 trace
                    write_host_properties(rank, pid, hostname);
                    // add the rank to the communicator group
                    group_members.push_back(rank);
                    /*
                    // add threads of the rank to the communicator location group
                    for (int i = 0 ; i < rank_thread_map[rank] ; i++) {
                        group_members_threads.push_back((group_members[rank] << 32) + i);
                    }
                    */
                    group_members_threads.push_back((group_members[rank] << 32));
                }
				std::cout << "Writing OTF2 Communicators..." << std::endl;
                // create the map of locations
                const char * world_locations = "MPI_COMM_WORLD_LOCATIONS";
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world_locations), world_locations ));
                OTF2_EC(OTF2_GlobalDefWriter_WriteGroup ( global_def_writer,
                    0, get_string_index(world_locations), OTF2_GROUP_TYPE_COMM_LOCATIONS,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE, group_members_threads.size(),
                    &group_members_threads[0]));   
                // create the map of ranks in the communicator - these have to be consectutive ranks.
                const char * world_group = "MPI_COMM_WORLD_GROUP";
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world_group), world_group ));
                OTF2_EC(OTF2_GlobalDefWriter_WriteGroup ( global_def_writer,
                    1, get_string_index(world_group), OTF2_GROUP_TYPE_COMM_GROUP,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE, group_members.size(),
                    &group_members[0]));   
                // create the communicator
                const char * world = "MPI_COMM_WORLD";
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world), world ));
                OTF2_EC(OTF2_GlobalDefWriter_WriteComm  ( global_def_writer,
                    0, get_string_index(world), 
                    1, OTF2_UNDEFINED_COMM));
            } else {
                // not rank 0? 
                // write out the timer names we saw
                write_my_regions();
                // write out the counter names we saw
                write_my_metrics();
                // using the reduced set of regions, write our local map
                // to the global strings
                write_region_map();
                write_metric_map();
            }
            for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
                /* close (and possibly create) the definition files */
                if (event_file_exists(i)) {
                    OTF2_EC(OTF2_Archive_CloseDefWriter( archive, getDefWriter(i) ));
                }
            }
			std::cout << "Closing the archive..." << std::endl;
            // close the archive! we are done!
            OTF2_EC(OTF2_Archive_Close( archive ));
            // delete our temporary files!
            // Commented out until we can figure out how to do this safely.
            /*
            if (my_saved_node_id == 0) {
                std::remove(otf2_listener::index_filename.c_str());
                ostringstream tmp;
                tmp << region_filename_prefix << "0";
                std::remove(tmp.str().c_str());
                ostringstream tmp2;
                tmp2 << metric_filename_prefix << "0";
                std::remove(tmp2.str().c_str());
            }
            */
			std::cout << "done." << std::endl;
        }
        return;
    }
    
    /* We need to check in with locality/rank 0 to let
     * it know how many localities/ranks there are in
     * the job. We do that by writing our rank to the 
     * master rank file (assuming a shared filesystem)
     * if it is larger than the current rank in there. */
    bool otf2_listener::write_my_node_properties() {
        // make sure we only call this function once
        static bool already_written = false;
        if (already_written) return true;
        // get our rank/locality info
        pid_t pid = ::getpid();
        char hostname[128];
        gethostname(hostname, sizeof hostname);
        string host(hostname);
        // build a string to write to the file
        stringstream ss;
        ss << my_saved_node_id << "\t" << pid << "\t" << hostname << "\n";
        cout << ss.str();
        string tmp = ss.str();
        std::ofstream index_file(index_filename + to_string(my_saved_node_id));
        // write our info
        index_file << tmp.c_str();
        // close the file
        index_file.close();
        already_written = true;
        return already_written;
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        return;
    }

    void otf2_listener::on_new_thread(new_thread_event_data &data) {
        /* the event writer and def writers are created using
         * static construction in on_start and on_stop */
        if (!thread_instance::map_id_to_worker(thread_instance::get_id())) {
            // don't close the archive on us!
            read_lock_type lock(_archive_mutex);
            // not likely, but just in case...
            if (_terminate) { return; }
            // before we process the event, make sure the event write is open
            OTF2_EvtWriter* local_evt_writer = getEvtWriter();
            if (local_evt_writer != NULL) {
              // insert a thread begin event
              uint64_t stamp = get_time();
              OTF2_EvtWriter_ThreadBegin( local_evt_writer, NULL, stamp, 0, thread_instance::get_id() );
            }
        }
        APEX_UNUSED(data);
        return;
    }

    void otf2_listener::on_exit_thread(event_data &data) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return; }
        //if (thread_instance::get_id() != 0) {
            // insert a thread end event
            uint64_t stamp = get_time();
            OTF2_EvtWriter_ThreadEnd( getEvtWriter(), NULL, stamp, 0, 0);
        //} 
        printf("closing event writer for thread %lu\n", thread_instance::get_id()); fflush(stdout);
        //_event_threads.insert(thread_instance::get_id());
        OTF2_Archive_CloseEvtWriter( archive, getEvtWriter() );
        if (thread_instance::get_id() == 0) {
            comm_evt_writer = nullptr;
        }
        APEX_UNUSED(data);
        return;
    }

    bool otf2_listener::on_start(task_identifier * id) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return false; }
        // before we process the event, make sure the event write is open
        OTF2_EvtWriter* local_evt_writer = getEvtWriter();
        if (thread_instance::get_id() == 0) {
            uint64_t idx = get_region_index(id);
            // Because the event writer for thread 0 is also
            // used for communication events and sampled values,
            // we have to get a lock for it.
            std::unique_lock<std::mutex> lock(_comm_mutex);
            // unfortunately, we can't use the timestamp from the
            // profiler object. bummer. it has to be taken after
            // the lock is acquired, so that events happen on
            // thread 0 in monotonic order.
            uint64_t stamp = get_time();
            OTF2_EC(OTF2_EvtWriter_Enter( local_evt_writer, NULL, stamp, idx /* region */ ));
        } else {
            uint64_t stamp = get_time();
            OTF2_EC(OTF2_EvtWriter_Enter( local_evt_writer, NULL, stamp, get_region_index(id) /* region */ ));
        }
        return true;
    }

    bool otf2_listener::on_resume(task_identifier * id) {
        return on_start(id);
    }

    void otf2_listener::on_stop(std::shared_ptr<profiler> &p) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        OTF2_EvtWriter* local_evt_writer = getEvtWriter();
        // not likely, but just in case...
        if (_terminate) { return; }
        if (thread_instance::get_id() == 0) {
            uint64_t idx = get_region_index(p->task_id);
            // Because the event writer for thread 0 is also
            // used for communication events and sampled values,
            // we have to get a lock for it.
            std::unique_lock<std::mutex> lock(_comm_mutex);
            // unfortunately, we can't use the timestamp from the
            // profiler object. bummer. it has to be taken after
            // the lock is acquired, so that events happen on
            // thread 0 in monotonic order.
            uint64_t stamp = get_time();
            OTF2_EC(OTF2_EvtWriter_Leave( local_evt_writer, NULL, stamp, idx /* region */ ));
        } else {
            uint64_t stamp = get_time();
            OTF2_EC(OTF2_EvtWriter_Leave( local_evt_writer, NULL, stamp, 
                    get_region_index(p->task_id) /* region */ ));
        }
        return;
    }

    void otf2_listener::on_yield(std::shared_ptr<profiler> &p) {
        on_stop(p);
    }

    /* The send is always assumed to be done by thread 0 */
    void otf2_listener::on_send(message_event_data &data) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return; }
        if (comm_evt_writer != NULL) {
            // create an empty attribute list. could be null?
            OTF2_AttributeList * attributeList = OTF2_AttributeList_New();
            // only one communicator, so hard coded.
            OTF2_CommRef communicator = 0;
            {
                // because we are writing to thread 0's event stream,
                // set the lock
                std::unique_lock<std::mutex> lock(_comm_mutex);
                // we have to get a timestamp after the lock, to make sure
                // that time stamps are monotonically increasing. :(
                uint64_t stamp = get_time();
                // write our recv into the event stream
                OTF2_EC(OTF2_EvtWriter_MpiSend  ( comm_evt_writer,
                        attributeList, stamp, data.target, communicator,
                        data.tag, data.size ));
            }
            OTF2_EC(OTF2_AttributeList_Delete(attributeList));
        }
        return;
    }

    /* The receive is always assumed to be done by thread 0,
     * because the sender doesnt' know which thread will handle
     * the parcel. But the receiver knows the sending thread. */
    void otf2_listener::on_recv(message_event_data &data) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return; }
        if (comm_evt_writer != NULL) {
            // create an empty attribute list. could be null?
            OTF2_AttributeList * attributeList = OTF2_AttributeList_New();
            // only one communicator, so hard coded.
            OTF2_CommRef communicator = 0;
            {
                // because we are writing to thread 0's event stream,
                // set the lock
                std::unique_lock<std::mutex> lock(_comm_mutex);
                // we have to get a timestamp after the lock, to make sure
                // that time stamps are monotonically increasing. :(
                uint64_t stamp = get_time();
                // write our recv into the event stream
                //std::cout << "receiving from: " << data.source_thread << std::endl;
                OTF2_EC(OTF2_EvtWriter_MpiRecv  ( comm_evt_writer,
                        attributeList, stamp, data.source_rank, communicator,
                        data.tag, data.size ));
            }
            // delete the attribute.
            OTF2_EC(OTF2_AttributeList_Delete(attributeList));
        }
        return;
    }

    void otf2_listener::on_sample_value(sample_value_event_data &data) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return; }
        // create a union for storing the value
        OTF2_MetricValue omv[1];
        omv[0].floating_point = data.counter_value;
        // tell the union what type this is
        OTF2_Type omt[1];
        omt[0]=OTF2_TYPE_DOUBLE;
        {
            uint64_t idx = get_metric_index(*(data.counter_name));
            // because we are writing to thread 0's event stream,
            // set the lock
            std::unique_lock<std::mutex> lock(_comm_mutex);
            // we have to get a timestamp after the lock, to make sure
            // that time stamps are monotonically increasing. :(
            uint64_t stamp = get_time();
            // write our counter into the event stream
            if (comm_evt_writer != NULL) {
                OTF2_EC(OTF2_EvtWriter_Metric( comm_evt_writer, NULL, stamp, idx, 1, omt, omv ));
            }
        }
        return;
    }
}
