//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "otf2_listener.hpp"
#include "thread_instance.hpp"
#include <sstream>
#include <ostream>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <ios>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#if defined(APEX_HAVE_MPI)
#include <mpi.h>
#endif
#include <atomic>

#define OTF2_EC(call) { \
    OTF2_ErrorCode ec = call; \
    if (ec != OTF2_SUCCESS) { \
        printf("OTF2 Error: %s, %s\n", OTF2_Error_GetName(ec),\
        OTF2_Error_GetDescription (ec)); \
    } \
}

using namespace std;

namespace apex {

    uint32_t otf2_listener::make_vtid (uint32_t device, uint32_t context, uint32_t stream) {
        cuda_thread_node tmp(device, context, stream);
        size_t tid;
        /* There is a potential for overlap here, but not a high potential.  The CPU and the GPU
        * would BOTH have to spawn 64k+ threads/streams for this to happen. */
        if (vthread_map.count(tmp) == 0) {
            // build the thread name for viewers
            std::stringstream ss;
            ss << "GPU Dev: " << device;
            if (context > 0) {
                ss << std::setfill('0');
                ss << " Ctx:";
                ss << std::setw(2) << context;
                if (stream > 0) {
                    ss << " Str:";
                    ss << setw(5) << stream;
                }
            }
            std::string name{ss.str()};
            // lock the archive lock, we need to make an event writer
            write_lock_type lock(_archive_mutex);
            // lock the set of thread IDs
            _event_set_mutex.lock();
            // get the next ID
            uint32_t id = (uint32_t)_event_threads.size();
            // reverse it, so as to avoid collisions with CPU threads
            // uint32_t id_reversed = simple_reverse(id);
            // insert it.
            //std::cout << "GPU Inserting " << _event_threads.size() << std::endl;
            size_t tmpid = _event_threads.size();
            _event_threads.insert(tmpid);
            _event_thread_names.insert(std::pair<uint32_t, std::string>(tmpid, name));
            // done with the set of event threads, so unlock.
            _event_set_mutex.unlock();
            // use the OTF2 thread index (not reversed) for the vthread_map
            vthread_map.insert(std::pair<cuda_thread_node, size_t>(tmp,id));
            // construct a globally unique ID for this thread on this rank
            uint64_t my_node_id = my_saved_node_id;
            my_node_id = (my_node_id << 32) + id;
            // construct the event writer
            OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter( archive, my_node_id );
            // add it to the map of virtual thread IDs to event writers
            vthread_evt_writer_map.insert(std::pair<size_t, OTF2_EvtWriter*>(id, evt_writer));
        }
        tid = vthread_map[tmp];
        return tid;
    }

    OTF2_FlushCallbacks otf2_listener::flush_callbacks;

    /* Stupid Intel compiler CLAIMS to be C++14, but doesn't have support for
     * std::unique_ptr. */
#if __cplusplus < 201402L || defined(__INTEL_COMPILER)
    template<typename T, typename... Args>
    std::unique_ptr<T> my_make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
#define APEX_MAKE_UNIQUE my_make_unique
#else
#define APEX_MAKE_UNIQUE std::make_unique
#endif

    struct string_sort_by_value {
        bool operator()(const std::pair<std::string,int> &left, const
            std::pair<std::string,int> &right) {
            return left.second < right.second;
        }
    };

    uint64_t otf2_listener::globalOffset(0);
    const std::string otf2_listener::empty("");
    int otf2_listener::my_saved_node_id(0);
    int otf2_listener::my_saved_node_count(1);
    std::atomic<int> active_threads{0};

    OTF2_CallbackCode otf2_listener::my_OTF2GetSize(void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *size) {
        /* Returns the number of OTF2_Archive objects operating in this
           communication context. */
        //cout << __func__ << " " << apex_options::otf2_collective_size() << endl;
        //*size = apex_options::otf2_collective_size();
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        *size = my_saved_node_count;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2GetRank (void *userData,
            OTF2_CollectiveContext *commContext, uint32_t *rank) {
        /* Returns the rank of this OTF2_Archive objects in this communication
           context. A number between 0 and one less of the size of the communication
           context. */
        //cout << __func__ << " " << my_saved_node_id << endl;
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(rank);
        *rank = my_saved_node_id;
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2CreateLocalComm (void *userData,
            OTF2_CollectiveContext **localCommContext, OTF2_CollectiveContext
            *globalCommContext, uint32_t globalRank, uint32_t globalSize, uint32_t
            localRank, uint32_t localSize, uint32_t fileNumber, uint32_t
            numberOfFiles) {
        /* Create a new disjoint partitioning of the the globalCommContext
           communication context. numberOfFiles denotes the number of the partitions.
           fileNumber denotes in which of the partitions this OTF2_Archive should belong.
           localSize is the size of this partition and localRank the rank of this
           OTF2_Archive in the partition. */
        APEX_UNUSED(userData);
        APEX_UNUSED(localCommContext);
        APEX_UNUSED(globalCommContext);
        APEX_UNUSED(globalRank);
        APEX_UNUSED(globalSize);
        APEX_UNUSED(localRank);
        APEX_UNUSED(localSize);
        APEX_UNUSED(fileNumber);
        APEX_UNUSED(numberOfFiles);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2FreeLocalComm (void *userData,
            OTF2_CollectiveContext *localCommContext) {
        /* Destroys the communication context previous created by the
           OTF2CreateLocalComm callback. */
        APEX_UNUSED(userData);
        APEX_UNUSED(localCommContext);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Barrier (void *userData,
            OTF2_CollectiveContext *commContext) {
        /* Performs a barrier collective on the given communication context. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Bcast (void *userData,
            OTF2_CollectiveContext *commContext, void *data, uint32_t numberElements,
            OTF2_Type type, uint32_t root) {
        /* Performs a broadcast collective on the given communication context. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(data);
        APEX_UNUSED(numberElements);
        APEX_UNUSED(type);
        APEX_UNUSED(root);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gather (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute the same number of elements. outData is only valid at
           rank root. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(inData);
        APEX_UNUSED(outData);
        APEX_UNUSED(numberElements);
        APEX_UNUSED(type);
        APEX_UNUSED(root);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Gatherv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, uint32_t inElements,
            void *outData, const uint32_t *outElements, OTF2_Type type, uint32_t root) {
        /* Performs a gather collective on the given communication context where
           each ranks contribute different number of elements. outData and outElements are
           only valid at rank root. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(inData);
        APEX_UNUSED(inElements);
        APEX_UNUSED(outData);
        APEX_UNUSED(outElements);
        APEX_UNUSED(type);
        APEX_UNUSED(root);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatter (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, void *outData,
            uint32_t numberElements, OTF2_Type type, uint32_t root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute the same number of elements. inData is only valid at rank
           root. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(inData);
        APEX_UNUSED(outData);
        APEX_UNUSED(numberElements);
        APEX_UNUSED(type);
        APEX_UNUSED(root);
        return OTF2_CALLBACK_SUCCESS;
    }

    OTF2_CallbackCode otf2_listener::my_OTF2Scatterv (void *userData,
            OTF2_CollectiveContext *commContext, const void *inData, const uint32_t
            *inElements, void *outData, uint32_t outElements, OTF2_Type type, uint32_t
            root) {
        /* Performs a scatter collective on the given communication context where
           each ranks contribute different number of elements. inData and inElements are
           only valid at rank root. */
        APEX_UNUSED(userData);
        APEX_UNUSED(commContext);
        APEX_UNUSED(inData);
        APEX_UNUSED(inElements);
        APEX_UNUSED(outData);
        APEX_UNUSED(outElements);
        APEX_UNUSED(type);
        APEX_UNUSED(root);
        return OTF2_CALLBACK_SUCCESS;
    }

    void otf2_listener::my_OTF2Release (void *userData, OTF2_CollectiveContext
            *globalCommContext, OTF2_CollectiveContext *localCommContext) {
        APEX_UNUSED(userData);
        APEX_UNUSED(globalCommContext);
        APEX_UNUSED(localCommContext);
        /* Optionally called in OTF2_Archive_Close or OTF2_Reader_Close
           respectively. */
        return;
    }

        /* these indices are thread-specific. */
        std::map<task_identifier,uint64_t>& otf2_listener::get_region_indices(void) {
            static APEX_NATIVE_TLS std::map<task_identifier,uint64_t> * region_indices;
            if (region_indices == nullptr) {
                region_indices = new std::map<task_identifier,uint64_t>();
            }
            return *region_indices;
        }
        /* these indices are thread-specific. */
        std::map<std::string,uint64_t>& otf2_listener::get_string_indices(void) {
            static APEX_NATIVE_TLS std::map<std::string,uint64_t> * string_indices;
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
            static APEX_NATIVE_TLS std::map<std::string,uint64_t> * metric_indices;
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

    OTF2_EvtWriter* otf2_listener::getEvtWriter(bool create) {
      static APEX_NATIVE_TLS OTF2_EvtWriter* evt_writer(nullptr);
      // Are we creating / opening an event writer?
      if (evt_writer == nullptr && create) {
        // should already be locked by the "new thread" event.
        uint64_t my_node_id = my_saved_node_id;
        std::unique_lock<std::mutex> l(_event_set_mutex);
        //my_node_id = (my_node_id << 32) + thread_instance::get_id();
        my_node_id = (my_node_id << 32) + _event_threads.size();
        evt_writer = OTF2_Archive_GetEvtWriter( archive, my_node_id );
        if (thread_instance::get_id() == 0) {
            comm_evt_writer = evt_writer;
        }
        //_event_threads.insert(thread_instance::get_id());
        //std::cout << "CPU Inserting " << _event_threads.size() << std::endl;
        size_t tmpid = _event_threads.size();
        _event_threads.insert(tmpid);
        // construct and save the name of this thread
        std::stringstream ss;
        ss << "CPU thread " << thread_instance::get_id();
        std::string name{ss.str()};
        _event_thread_names.insert(std::pair<uint32_t, std::string>(tmpid, name));
      // Are we closing an event writer?
      } else if (!create) {
        if (thread_instance::get_id() == 0) {
            comm_evt_writer = nullptr;
        }
        if (evt_writer != nullptr) {
            //printf("closing event writer %p for thread %lu\n", evt_writer,
            //thread_instance::get_id()); fflush(stdout);
            // FOR SOME REASON, OTF2 CRASHES ON EXIT
            // Not closing the event writers seems to prevent that?
            // OTF2_Archive_CloseEvtWriter( archive, evt_writer );
            evt_writer = nullptr;
        }
      }
      //printf("using event writer %p for thread %lu\n", evt_writer,
      //thread_instance::get_id()); fflush(stdout);
      return evt_writer;
    }

    bool otf2_listener::event_file_exists (uint32_t threadid) {
        // get exclusive access to the set - unlocks on exit
        std::unique_lock<std::mutex> l(_event_set_mutex);
        if (_event_threads.find(threadid) == _event_threads.end())
        {return false;} else {return true;}
    }

    OTF2_DefWriter* otf2_listener::getDefWriter(uint32_t threadid) {
        OTF2_DefWriter* def_writer;
        //printf("creating definition writer for thread %d\n", threadid);
        //fflush(stdout);
        uint64_t my_node_id = my_saved_node_id;
        my_node_id = (my_node_id << 32) + threadid;
        def_writer = OTF2_Archive_GetDefWriter( archive, my_node_id );
        return def_writer;
    }

    /* constructor for the OTF2 listener class */
    otf2_listener::otf2_listener (void) : _terminate(false),
        _initialized(false), comm_evt_writer(nullptr),
        global_def_writer(nullptr), dropped(0) {
        /* get a start time for the trace */
        globalOffset = get_time();
        /* set the flusher */
        flush_callbacks.otf2_pre_flush  = otf2_listener::pre_flush;
        flush_callbacks.otf2_post_flush = otf2_listener::post_flush;
        index_filename = string(string(apex_options::otf2_archive_path())
            + "/.locality.");
        region_filename_prefix =
            string(string(apex_options::otf2_archive_path()) + "/.regions.");
        metric_filename_prefix =
            string(string(apex_options::otf2_archive_path()) + "/.metrics.");
        thread_filename_prefix =
            string(string(apex_options::otf2_archive_path()) + "/.threads.");
        lock_filename_prefix =
            string(string(apex_options::otf2_archive_path()) +
            "/.regions.lock.");
        lock2_filename_prefix =
            string(string(apex_options::otf2_archive_path()) +
            "/.metrics.lock.");
        lock3_filename_prefix =
            string(string(apex_options::otf2_archive_path()) +
            "/.threads.lock.");
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
            std::cout << "removing path!" << std::endl; fflush(stdout);
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
        OTF2_EC(OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks,
            nullptr ));
        /* set the creator name */
        stringstream tmp;
        tmp << "APEX version " << version();
        OTF2_EC(OTF2_Archive_SetCreator(archive, tmp.str().c_str()));
        /* we have no collective callbacks. */
        OTF2_EC(OTF2_Archive_SetCollectiveCallbacks(archive,
            get_collective_callbacks(), nullptr, nullptr, nullptr));
        /* open the event files for this archive */
        OTF2_EC(OTF2_Archive_OpenEvtFiles( archive ));
        created = true;
        return created;
     }

    void otf2_listener::on_startup(startup_event_data &data) {
        APEX_UNUSED(data);
       // add the empty string to the string definitions
        get_string_index(empty);
        // save the node id, because the apex object my not be
        // around when we are finalizing everything.
        my_saved_node_id = apex::instance()->get_node_id();
        my_saved_node_count = apex::instance()->get_num_ranks();
        cout << "Rank " << my_saved_node_id << " of " << my_saved_node_count <<
            "." << endl;
        // now is a good time to make sure the archive is open on this
        // rank/locality
        static bool archive_created = create_archive();
        if ((!_terminate) && archive_created) {
            // set up the event writer for communication (thread 0).
            getEvtWriter(true);
        } else {
            std::cerr << "Archive not created!" << std::endl; fflush(stderr);
            return;
        }
        _initialized = true;
        return;
    }

    void otf2_listener::write_otf2_attributes(void) {
        // only write these out once!
        static APEX_NATIVE_TLS bool written = false;
        if (written) return;
        written = true;
        // write the GUID attribute
        const char * name = "GUID";
        const char * description = "Globaly unique identifier";
        OTF2_StringRef name_ref = get_string_index(name);
        OTF2_StringRef desc_ref = get_string_index(description);
        OTF2_GlobalDefWriter_WriteString( global_def_writer, name_ref, name );
        OTF2_GlobalDefWriter_WriteString( global_def_writer, desc_ref,
            description );
        OTF2_AttributeRef guid_ref = 0;
        OTF2_GlobalDefWriter_WriteAttribute( global_def_writer,
                guid_ref, name_ref, desc_ref, OTF2_TYPE_UINT64);
        // write the parent GUID attribute
        const char * parent_name = "Parent GUID";
        const char * parent_description =
            "Globaly unique identifier of the parent task";
        OTF2_StringRef parent_name_ref = get_string_index(parent_name);
        OTF2_StringRef parent_desc_ref = get_string_index(parent_description);
        OTF2_GlobalDefWriter_WriteString( global_def_writer, parent_name_ref,
            parent_name );
        OTF2_GlobalDefWriter_WriteString( global_def_writer, parent_desc_ref,
            parent_description );
        OTF2_AttributeRef parent_guid_ref = 1;
        OTF2_GlobalDefWriter_WriteAttribute( global_def_writer,
            parent_guid_ref, parent_name_ref, parent_desc_ref, OTF2_TYPE_UINT64);
    }

    inline void convert_upper(std::string& str, std::string& converted)
    {
        for(size_t i = 0; i < str.size(); ++i) {
            converted += toupper(str[i]);
        }
    }

    void otf2_listener::write_otf2_regions(void) {
        // only write these out once!
        static APEX_NATIVE_TLS bool written = false;
        if (written) return;
        written = true;
        // need to sort the map by ID, not name.
        std::map<uint64_t,std::string> sorted_reduced_region_map;
        for (auto const &i : reduced_region_map) {
            sorted_reduced_region_map[i.second] = i.first;
        }

        for (auto const &i : sorted_reduced_region_map) {
            string id = i.second;
            uint64_t idx = i.first;
            OTF2_GlobalDefWriter_WriteString( global_def_writer,
                get_string_index(id), id.c_str() );
            OTF2_Paradigm paradigm = OTF2_PARADIGM_USER;
            OTF2_RegionRole role = OTF2_REGION_ROLE_TASK;
            string uppercase;
            convert_upper(id, uppercase);
            // does the original string contain APEX?
            size_t found = id.find(string("APEX"));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_MEASUREMENT_SYSTEM;
                role = OTF2_REGION_ROLE_ARTIFICIAL;
            }
            found = uppercase.find(string("UNRESOLVED"));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_MEASUREMENT_SYSTEM;
                role = OTF2_REGION_ROLE_ARTIFICIAL;
            }
            found = uppercase.find(string("OPENMP"));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_OPENMP;
                role = OTF2_REGION_ROLE_WRAPPER;
            }
            found = uppercase.find(string("PTHREAD"));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_PTHREAD;
            }
            found = uppercase.find(string("MPI"));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_MPI;
                role = OTF2_REGION_ROLE_WRAPPER;
            }
            // does the original string start with GPU:?
            found = id.find(string("GPU: "));
            if (found != std::string::npos) {
                paradigm = OTF2_PARADIGM_CUDA;
                role = OTF2_REGION_ROLE_FUNCTION;
                found = id.find(string("Memory copy"));
                if (found != std::string::npos) {
                    role = OTF2_REGION_ROLE_DATA_TRANSFER;
                }
            }
            // does the original string start with cuda?
            found = id.rfind(string("cuda"),0);
            if (found == 0) {
                paradigm = OTF2_PARADIGM_CUDA;
                role = OTF2_REGION_ROLE_WRAPPER;
                found = uppercase.find(string("MEMCPY"));
                if (found != std::string::npos) {
                    role = OTF2_REGION_ROLE_DATA_TRANSFER;
                }
                found = uppercase.find(string("SYNC"));
                if (found != std::string::npos) {
                    role = OTF2_REGION_ROLE_TASK_WAIT;
                }
                found = uppercase.find(string("MALLOC"));
                if (found != std::string::npos) {
                    role = OTF2_REGION_ROLE_ALLOCATE;
                }
                found = uppercase.find(string("FREE"));
                if (found != std::string::npos) {
                    role = OTF2_REGION_ROLE_DEALLOCATE;
                }
            }
            OTF2_GlobalDefWriter_WriteRegion( global_def_writer,
                    idx /* id */,
                    get_string_index(id) /* region name  */,
                    get_string_index(empty) /* alternative name */,
                    get_string_index(empty) /* description */,
                    role,
                    paradigm,
                    OTF2_REGION_FLAG_NONE,
                    get_string_index(empty) /* source file */,
                    get_string_index(empty) /* begin lno */,
                    get_string_index(empty) /* end lno */ );
        }
    }

    void otf2_listener::write_otf2_metrics(void) {
        // only write these out once!
        static APEX_NATIVE_TLS bool written = false;
        if (written) return;
        written = true;
        // write a "unit" string
        OTF2_GlobalDefWriter_WriteString( global_def_writer,
            get_string_index("count"), "count" );
        // copy the reduced map to a pair, so we can sort by value
        std::vector<std::pair<std::string, int>> pairs;
        for (auto const &i : reduced_metric_map) {
            pairs.push_back(i);
        }
        sort(pairs.begin(), pairs.end(), string_sort_by_value());
        // iterate over the metrics and write them out.
        for (auto const &i : pairs) {
            string id = i.first;
            uint64_t idx = i.second;
            OTF2_GlobalDefWriter_WriteString( global_def_writer,
                get_string_index(id), id.c_str() );
            OTF2_GlobalDefWriter_WriteMetricMember( global_def_writer,
                idx, get_string_index(id), get_string_index(id),
                OTF2_METRIC_TYPE_OTHER, OTF2_METRIC_ABSOLUTE_POINT,
                OTF2_TYPE_DOUBLE, OTF2_BASE_DECIMAL, 0,
                get_string_index("count"));
            OTF2_MetricMemberRef omr[1];
            omr[0]=idx;
            OTF2_GlobalDefWriter_WriteMetricClass( global_def_writer,
                    idx, 1, omr, OTF2_METRIC_ASYNCHRONOUS,
                    OTF2_RECORDER_KIND_UNKNOWN);
        }
    }

    void otf2_listener::write_region_map(std::map<std::string,uint64_t>&
        reduced_region_map) {
        // build the array of uint64_t values
        if (global_region_indices.size() > 0) {
            uint64_t * mappings = (uint64_t*)(malloc(sizeof(uint64_t) *
                global_region_indices.size()));
            for (auto const &i : global_region_indices) {
                task_identifier id = i.first;
                uint64_t idx = i.second;
                uint64_t mapped_index = reduced_region_map[id.get_name()];
                mappings[idx] = mapped_index;
                if (my_saved_node_id > 0) {
                /* Debug output
                std::cout << my_saved_node_id
                          << " idx: " << idx
                          << " mapped_index: " << mapped_index
                          << " id: " << id.get_name()
                          << std::endl;
                 */
                }
            }
            // create a map
            uint64_t map_size = global_region_indices.size();
            OTF2_IdMap * my_map = OTF2_IdMap_CreateFromUint64Array(map_size,
                mappings, false);
            for (size_t i = 0 ; i < _event_threads.size() ; i++) {
                OTF2_DefWriter_WriteMappingTable(getDefWriter(i),
                    OTF2_MAPPING_REGION, my_map);
            }
            // free the map
            OTF2_IdMap_Free(my_map);
            free(mappings);
        }
    }

    void otf2_listener::write_metric_map(std::map<std::string,uint64_t>&
        reduced_metric_map) {
        // build the array of uint64_t values
        if (global_metric_indices.size() > 0) {
            uint64_t * mappings = (uint64_t*)(malloc(sizeof(uint64_t) *
                global_metric_indices.size()));
            for (auto const &i : global_metric_indices) {
                string name = i.first;
                uint64_t idx = i.second;
                uint64_t mapped_index = reduced_metric_map[name];
                mappings[idx] = mapped_index;
            }
            // create a map
            uint64_t map_size = global_metric_indices.size();
            OTF2_IdMap * my_map = OTF2_IdMap_CreateFromUint64Array(map_size,
                mappings, false);
            for (size_t i = 0 ; i < _event_threads.size() ; i++) {
                OTF2_DefWriter_WriteMappingTable(getDefWriter(i),
                    OTF2_MAPPING_METRIC, my_map);
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
    void otf2_listener::write_host_properties(int rank, int pid, std::string&
        hostname) {
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
        //for (int i = 0 ; i < rank_thread_name_map[rank] ; i++) {
        for (auto iter : rank_thread_name_map[rank]) {
            uint32_t index = iter.first;
            std::string name = iter.second;
            uint64_t thread_id = node_id + index;
            // have we written this thread name before?
            auto tmp = threadnames.find(name);
            if (tmp == threadnames.end()) {
                OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(name), name.c_str() );
                threadnames.insert(name);
            }
            OTF2_LocationType lt = OTF2_LOCATION_TYPE_CPU_THREAD;
            if (name.rfind("GPU ",0) == 0) {
                lt = OTF2_LOCATION_TYPE_GPU;
            }
            // write out the thread location into the system tree
            OTF2_GlobalDefWriter_WriteLocation( global_def_writer,
                thread_id /* id */,
                get_string_index(name) /* name */,
                lt,
                rank_region_map[rank] /* number of events */,
                rank /* location group ID */ );
        }
    }

    /* do nothing with OTF2 at dump event. For now. */
    void otf2_listener::on_dump(dump_event_data &data) {
        APEX_UNUSED(data);
        return;
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
            //std::cout << "Waiting for all support threads to exit..." << std::endl;
            usleep(apex_options::policy_drain_timeout()); // sleep 1ms (default)
            /* close event files */
            if (my_saved_node_id == 0) {
                std::cout << "Closing OTF2 event files..." << std::endl;
            }
            OTF2_EC(OTF2_Archive_CloseEvtFiles( archive ));
            // write node properties
            auto map_pair = reduce_node_properties(write_my_node_properties());
            /* if we are node 0, write the global definitions */
            if (my_saved_node_id == 0) {
                // save my number of threads
                //rank_thread_map[0] = thread_instance::get_num_threads();
                rank_thread_map[0] = _event_threads.size();
                std::cout << "Writing OTF2 definition files..." << std::endl;
                // make a common list of regions and metrics across all nodes...
                reduce_regions();
                reduce_metrics();
                reduce_threads();
                std::cout << "Writing OTF2 Global definition file..." << std::endl;
                // create the global definition writer
                global_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );
                // write an "empty" string - only once
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(empty), empty.c_str() ));
                // write out the reduced set of regions
                write_otf2_regions();
                // write out the common set of attributes
                write_otf2_attributes();
                // write out the reduced set of metrics
                write_otf2_metrics();
                // write out the clock properties
                write_clock_properties();
                // write a "node" string - only once
                const string node("node");
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(node), node.c_str() ));
                // let rank/locality 0 know this rank's properties.
                auto rank_pid_map = std::get<0>(*map_pair);
                auto rank_hostname_map = std::get<1>(*map_pair);
                // these are communicator lists, and a location map
                // for each. We need a group member for each process,
                // and the "location" is thread 0 of that process.
                vector<uint64_t>group_members;
                vector<uint64_t>group_members_threads;
                std::cout << "Writing OTF2 Node information..." << std::endl;
                // iterate over the ranks (in order) and write them out
                for (auto const &i : rank_pid_map) {
                    int rank = i.first;
                    int pid = i.second;
                    // write the host properties to the OTF2 trace
                    write_host_properties(rank, pid, rank_hostname_map[rank]);
                    // add the rank to the communicator group
                    group_members.push_back(rank);
                    /*
                    // add threads of the rank to the communicator location group
                    for (int i = 0 ; i < rank_thread_map[rank] ; i++) {
                        group_members_threads.push_back((group_members[rank] <<
                        32) + i);
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
                    0, get_string_index(world_locations),
                    OTF2_GROUP_TYPE_COMM_LOCATIONS,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE,
                    group_members_threads.size(),
                    &group_members_threads[0]));
                // create the map of ranks in the communicator - these have to
                // be consectutive ranks.
                const char * world_group = "MPI_COMM_WORLD_GROUP";
                OTF2_EC(OTF2_GlobalDefWriter_WriteString( global_def_writer,
                    get_string_index(world_group), world_group ));
                OTF2_EC(OTF2_GlobalDefWriter_WriteGroup ( global_def_writer,
                    1, get_string_index(world_group), OTF2_GROUP_TYPE_COMM_GROUP,
                    OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE,
                    group_members.size(), &group_members[0]));
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
                reduce_regions();
                // write out the counter names we saw
                reduce_metrics();
                // write out the thread names we saw
                reduce_threads();
            }
            //for (int i = 0 ; i < thread_instance::get_num_threads() ; i++) {
            for (size_t i = 0 ; i < _event_threads.size() ; i++) {
                /* close (and possibly create) the definition files */
                //if (event_file_exists(i)) {
                    OTF2_EC(OTF2_Archive_CloseDefWriter( archive,
                        getDefWriter(i) ));
                //}
            }
            if (my_saved_node_id == 0) {
                std::cout << "Closing the archive..." << std::endl;
            }
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
            if (my_saved_node_id == 0) {
                std::cout << "done." << std::endl;
            }
        }
        return;
    }

    /* We need to check in with locality/rank 0 to let
     * it know how many localities/ranks there are in
     * the job. We do that by writing our rank to the
     * master rank file (assuming a shared filesystem)
     * if it is larger than the current rank in there. */
    std::string otf2_listener::write_my_node_properties() {
        // get our rank/locality info
        pid_t pid = ::getpid();
        char hostname[128];
        gethostname(hostname, sizeof hostname);
        string host(hostname);
        // build a string to write to the file
        stringstream ss;
        ss << my_saved_node_id << "\t" << pid << "\t" << hostname << "\n";
        return ss.str();
    }

    void otf2_listener::on_new_node(node_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void otf2_listener::on_new_thread(new_thread_event_data &data) {
        /* the event writer and def writers are created using
         * static construction in on_start and on_stop */
            // don't close the archive on us!
            write_lock_type lock(_archive_mutex);
            // not likely, but just in case...
            if (_terminate) { return; }
            // before we process the event, make sure the event write is open
            getEvtWriter(true);
/* Don't do this until we can do the ThreadCreate call
 * and get the right threadContingent object for the
 * fourth parameter. */
#if 0
            OTF2_EvtWriter* local_evt_writer = getEvtWriter(true);
            if (local_evt_writer != nullptr) {
              // insert a thread begin event
              uint64_t stamp = get_time();
              OTF2_EvtWriter_ThreadBegin( local_evt_writer, nullptr, stamp, 0,
                thread_instance::get_id() );
            }
#endif
        APEX_UNUSED(data);
        return;
    }

    void otf2_listener::on_exit_thread(event_data &data) {
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return; }
/* Don't do this until we can do the ThreadCreate call
 * and get the right threadContingent object for the
 * fourth parameter. */
#if 0
        if (thread_instance::get_id() != 0) {
            // insert a thread end event
            uint64_t stamp = get_time();
            OTF2_EvtWriter_ThreadEnd( getEvtWriter(false), nullptr, stamp, 0, 0);
        }
#endif
        //_event_threads.insert(thread_instance::get_id());
        getEvtWriter(false);
        APEX_UNUSED(data);
        return;
    }

#if APEX_HAVE_PAPI
    void otf2_listener::write_papi_counters(OTF2_EvtWriter* writer, profiler*
        prof, uint64_t stamp, bool is_enter) {
        // create a union for storing the value
        OTF2_MetricValue omv[1];
        // tell the union what type this is
        OTF2_Type omt[1];
        omt[0]=OTF2_TYPE_DOUBLE;
        int i = 0;
        uint64_t idx = 0L;
        for (auto metric :
            apex::instance()->the_profiler_listener->get_metric_names()) {
            if (is_enter) {
                omv[0].floating_point = prof->papi_start_values[i++];
            } else {
                omv[0].floating_point = prof->papi_stop_values[i++];
            }
            idx = get_metric_index(metric);
            // write our counter into the event stream
            OTF2_EC(OTF2_EvtWriter_Metric( writer, nullptr, stamp, idx,
                1, omt, omv ));
        }
    }
#endif

    bool otf2_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
        // This could be a callback from a library before APEX is ready
        // Something like OpenMP or CUDA/CUPTI or...?
        if (!_initialized) return false;
        task_identifier * id = tt_ptr->get_task_id();
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // not likely, but just in case...
        if (_terminate) { return false; }
        // before we process the event, make sure the event write is open
        OTF2_EvtWriter* local_evt_writer = getEvtWriter(true);
        if (local_evt_writer != nullptr) {
            // create an attribute list
            OTF2_AttributeList * al = OTF2_AttributeList_New();
            // create an attribute
            OTF2_AttributeList_AddUint64( al, 0, tt_ptr->guid );
            OTF2_AttributeList_AddUint64( al, 1, tt_ptr->parent_guid );
            uint64_t idx = get_region_index(id);
            uint64_t stamp = 0L;
            if (thread_instance::get_id() == 0) {
                // Because the event writer for thread 0 is also
                // used for communication events and sampled values,
                // we have to get a lock for it.
                std::unique_lock<std::mutex> lock(_comm_mutex);
                // unfortunately, we can't use the timestamp from the
                // profiler object. bummer. it has to be taken after
                // the lock is acquired, so that events happen on
                // thread 0 in monotonic order.
                stamp = get_time();
                OTF2_EC(OTF2_EvtWriter_Enter( local_evt_writer, al,
                    stamp, idx /* region */ ));
#if APEX_HAVE_PAPI
                // write PAPI metrics!
                write_papi_counters(local_evt_writer, tt_ptr->prof,
                    stamp, true);
#endif
            } else {
                stamp = get_time();
                OTF2_EC(OTF2_EvtWriter_Enter( local_evt_writer, al,
                    stamp, idx /* region */ ));
#if APEX_HAVE_PAPI
                // write PAPI metrics!
                write_papi_counters(local_evt_writer, tt_ptr->prof,
                    stamp, true);
#endif
            }
            // delete the attribute list
            OTF2_AttributeList_Delete(al);
            return true;
        }
        return false;
    }

    bool otf2_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
        return on_start(tt_ptr);
    }

    void otf2_listener::on_stop(std::shared_ptr<profiler> &p) {
        // This could be a callback from a library before APEX is ready
        // Something like OpenMP or CUDA/CUPTI or...?
        if (!_initialized) return ;
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        OTF2_EvtWriter* local_evt_writer = getEvtWriter(true);
        if (local_evt_writer != nullptr) {
            // not likely, but just in case...
            if (_terminate) { return; }
            // create an attribute list
            OTF2_AttributeList * al = OTF2_AttributeList_New();
            // create an attribute
            OTF2_AttributeList_AddUint64( al, 0, p->tt_ptr->guid );
            OTF2_AttributeList_AddUint64( al, 1, p->tt_ptr->parent_guid );
            // unfortunately, we can't use the timestamp from the
            // profiler object. bummer. it has to be taken after
            // the lock is acquired, so that events happen on
            // thread 0 in monotonic order.
            uint64_t stamp = 0L;
            uint64_t idx = get_region_index(p->get_task_id());
            if (thread_instance::get_id() == 0) {
                // Because the event writer for thread 0 is also
                // used for communication events and sampled values,
                // we have to get a lock for it.
                std::unique_lock<std::mutex> lock(_comm_mutex);
                stamp = get_time();
                OTF2_EC(OTF2_EvtWriter_Leave( local_evt_writer, al,
                    stamp, idx /* region */ ));
#if APEX_HAVE_PAPI
                // write PAPI metrics!
                write_papi_counters(local_evt_writer, p.get(),
                    stamp, false);
#endif
            } else {
                stamp = get_time();
                OTF2_EC(OTF2_EvtWriter_Leave( local_evt_writer, al,
                    stamp, idx /* region */ ));
#if APEX_HAVE_PAPI
                // write PAPI metrics!
                write_papi_counters(local_evt_writer, p.get(),
                    stamp, false);
#endif
            }
            // delete the attribute list
            OTF2_AttributeList_Delete(al);
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
        if (comm_evt_writer != nullptr) {
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
        if (comm_evt_writer != nullptr) {
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
                //std::cout << "receiving from: " << data.source_thread <<
                //std::endl;
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
        // This could be an asynchronous sampled counter, may have gotten
        // here before initialization is done.  Wait a sec...hopefully not longer.
        while (!_initialized) {
            usleep(apex_options::policy_drain_timeout()); // sleep 1ms (default)
        }
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
            if (comm_evt_writer != nullptr) {
                OTF2_EC(OTF2_EvtWriter_Metric( comm_evt_writer, nullptr, stamp,
                idx, 1, omt, omv ));
            }
        }
        return;
    }

/* HPX implementation - TBD - for now, stick with file based reduction */
/*
#if defined(APEX_HAVE_HPX)


    std::unique_ptr<std::tuple<std::map<int,int>, std::map<int,std::string> > >
    otf2_listener::reduce_node_properties(std::string&& str) {
        return nullptr;
    }

#elif defined(APEX_HAVE_MPI)
*/

#if defined(APEX_HAVE_MPI)

    /* MPI implementations */

    std::unique_ptr<std::tuple<std::map<int,int>,
                    std::map<int,std::string> > >
         otf2_listener::reduce_node_properties(std::string&& str) {
        const int hostlength = 128;
        // get all hostnames
        char tmp[hostlength+1];
        strncpy(tmp, str.c_str(), hostlength);

        // make array for all hostnames
        char * allhostnames = nullptr;
        if (my_saved_node_id == 0) {
            allhostnames = (char*)calloc(hostlength*my_saved_node_count,
                sizeof(char));
        }

        if (my_saved_node_count > 1) {
            PMPI_Gather(tmp, hostlength, MPI_CHAR, allhostnames,
                        hostlength, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            allhostnames = &(tmp[0]);
        }

        // if not root, we are done.  return.
        if (my_saved_node_id > 0) {
            return nullptr;
        }

        std::map<int,int> rank_pid_map;
        std::map<int,string> rank_hostname_map;
        int rank, pid;
        std::string hostname;

        // point to the head of the array
        char * host_index = allhostnames;
        // find the lowest rank with my hostname
        for (int i = 0 ; i < my_saved_node_count ; i++) {
            char line[hostlength+1];
            strncpy(line, host_index, hostlength);
            istringstream ss(line);
            ss >> rank >> pid >> hostname;
            rank_pid_map[rank] = pid;
            rank_hostname_map[rank] = hostname;
            host_index = host_index + hostlength;
        }
        return APEX_MAKE_UNIQUE<std::tuple<std::map<int,int>,
                                std::map<int,string> > >(
                                rank_pid_map, rank_hostname_map);
    }

    std::string otf2_listener::write_my_regions(void) {
        stringstream region_file;
        // first, output our number of threads.
        //region_file << thread_instance::get_num_threads() << endl;
        region_file << _event_threads.size() << endl;
        // then iterate over the regions and write them out.
        for (auto const &i : global_region_indices) {
            task_identifier id = i.first;
            //uint64_t idx = i.second;
            //region_file << id.get_name() << "\t" << idx << endl;
            region_file << id.get_name() << endl;
        }
        return region_file.str();
    }

    int otf2_listener::reduce_regions(void) {
        std::string my_regions = write_my_regions();
        int length = my_regions.size();
        int full_length = 0;
        int max_length = 0;
        char * sbuf = nullptr;
        char * rbuf = nullptr;

        if (my_saved_node_count > 1) {
            // get the max length from all nodes
            PMPI_Allreduce(&length, &max_length, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD);
        } else {
            max_length = length;
        }
        // get the total length
        full_length = max_length * my_saved_node_count;

        // allocate space to store the strings on node 0
        if (my_saved_node_id == 0) {
            rbuf = (char*) calloc(full_length, sizeof(char));
        }
        // put the local data in a fixed length string
        sbuf = (char*) calloc(max_length, sizeof(char));
        strncpy(sbuf, my_regions.c_str(), max_length);

        if (my_saved_node_count > 1) {
            // gather the strings to node 0
            PMPI_Gather(sbuf, max_length, MPI_CHAR, rbuf,
                max_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            rbuf = sbuf;
        }

        int fullmap_length = 0;
        char * fullmap = nullptr;
        if (my_saved_node_id == 0) {
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
            for (int i = 1 ; i < my_saved_node_count ; i++) {
                comm_size++;
                rank_region_map[i] = 0;
                std::string region_file_string(rbuf+(max_length * i),
                    max_length);
                // get the number of threads from this rank
                std::string region_line;
                std::stringstream region_file(region_file_string);
                std::getline(region_file, region_line);
                std::string::size_type sz;   // alias of size_t
                rank_thread_map[i] = std::stoi(region_line,&sz);
                // read the map from that rank
                while (std::getline(region_file, region_line)) {
                    rank_region_map[i] = rank_region_map[i] + 1;
                    // trim the newline
                    region_line.erase(std::remove(region_line.begin(),
                        region_line.end(), '\n'), region_line.end());
                    if (reduced_region_map.find(region_line) ==
                        reduced_region_map.end()) {
                        uint64_t idx = reduced_region_map.size();
                        reduced_region_map[region_line] = idx;
                    }
                }
            }
            // copy the reduced map to a pair, so we can sort by value
            std::vector<std::pair<std::string, int>> pairs;
            for (auto const &i : reduced_region_map) {
                pairs.push_back(i);
            }
            sort(pairs.begin(), pairs.end(), string_sort_by_value());
            std::stringstream region_file;
            // iterate over the regions and build the string
            for (auto const &i : pairs) {
                std::string name = i.first;
                uint64_t idx = i.second;
                region_file << idx << "\t" << name << endl;
            }
            fullmap_length = region_file.str().length();
            fullmap = (char*) calloc(fullmap_length, sizeof(char));
            strncpy(fullmap, region_file.str().c_str(), fullmap_length);
        }

        if (my_saved_node_count > 1) {
            PMPI_Barrier(MPI_COMM_WORLD);
        }

        // share the full map length
        if (my_saved_node_count > 1) {
            PMPI_Bcast(&fullmap_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        if (my_saved_node_id > 0) {
            fullmap = (char*) calloc(fullmap_length, sizeof(char));
        }

        if (my_saved_node_count > 1) {
            // share the full map
            PMPI_Bcast(fullmap, fullmap_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        // read the reduced data
        if (my_saved_node_count > 1) {
            std::map<std::string,uint64_t> reduced_region_map;
            std::string region_line;
            std::stringstream region_file(fullmap);
            std::string region_name;
            int idx;
            // read the map from rank 0
            while (std::getline(region_file, region_line)) {
                istringstream ss(region_line);
                ss >> idx >> region_name;
                reduced_region_map[region_name] = idx;
            }
            // ...and write the map to the local definitions
            write_region_map(reduced_region_map);
        }
        return my_saved_node_count;
    }

    std::string otf2_listener::write_my_metrics(void) {
        stringstream metric_file;
        // first, output our number of threads.
        //metric_file << thread_instance::get_num_threads() << endl;
        metric_file << _event_threads.size() << endl;
        // then iterate over the metrics and write them out.
        for (auto const &i : global_metric_indices) {
            string id = i.first;
            //uint64_t idx = i.second;
            //metric_file << id.get_name() << "\t" << idx << endl;
            metric_file << id << endl;
        }
        return metric_file.str();
    }

    void otf2_listener::reduce_metrics(void) {
        std::string my_metrics = write_my_metrics();
        int length = my_metrics.size();
        int full_length = 0;
        int max_length = 0;
        char * sbuf = nullptr;
        char * rbuf = nullptr;

        if (my_saved_node_count > 1) {
            // get the max length from all nodes
            PMPI_Allreduce(&length, &max_length, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD);
        } else {
            max_length = length;
        }
        // get the total length
        full_length = max_length * my_saved_node_count;

        // allocate space to store the strings on node 0
        if (my_saved_node_id == 0) {
            rbuf = (char*) calloc(full_length, sizeof(char));
        }
        // put the local data in a fixed length string
        sbuf = (char*) calloc(max_length, sizeof(char));
        strncpy(sbuf, my_metrics.c_str(), max_length);

        if (my_saved_node_count > 1) {
            // gather the strings to node 0
            PMPI_Gather(sbuf, max_length, MPI_CHAR, rbuf,
                max_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            rbuf = sbuf;
        }

        int fullmap_length = 0;
        char * fullmap = nullptr;
        if (my_saved_node_id == 0) {
            // iterate over my metric map, and build a map of strings to ids
            // save my number of metrics
            rank_metric_map[0] = global_metric_indices.size();
            for (auto const &i : global_metric_indices) {
                task_identifier id = i.first;
                uint64_t idx = i.second;
                reduced_metric_map[id.get_name()] = idx;
            }
            int comm_size = 0;
            // iterate over the other ranks in the index files
            for (int i = 1 ; i < my_saved_node_count ; i++) {
                comm_size++;
                rank_metric_map[i] = 0;
                std::string metric_file_string(rbuf+(max_length * i),
                    max_length);
                // get the number of threads from this rank
                std::string metric_line;
                std::stringstream metric_file(metric_file_string);
                std::getline(metric_file, metric_line);
                std::string::size_type sz;   // alias of size_t
                rank_thread_map[i] = std::stoi(metric_line,&sz);
                // read the map from that rank
                while (std::getline(metric_file, metric_line)) {
                    rank_metric_map[i] = rank_metric_map[i] + 1;
                    // trim the newline
                    metric_line.erase(std::remove(metric_line.begin(),
                        metric_line.end(), '\n'), metric_line.end());
                    if (reduced_metric_map.find(metric_line) ==
                        reduced_metric_map.end()) {
                        uint64_t idx = reduced_metric_map.size();
                        reduced_metric_map[metric_line] = idx;
                    }
                }
            }
            // copy the reduced map to a pair, so we can sort by value
            std::vector<std::pair<std::string, int>> pairs;
            for (auto const &i : reduced_metric_map) {
                pairs.push_back(i);
            }
            sort(pairs.begin(), pairs.end(), string_sort_by_value());
            std::stringstream metric_file;
            // iterate over the metrics and build the string
            for (auto const &i : pairs) {
                std::string name = i.first;
                uint64_t idx = i.second;
                metric_file << idx << "\t" << name << endl;
            }
            fullmap_length = metric_file.str().length();
            fullmap = (char*) calloc(fullmap_length, sizeof(char));
            strncpy(fullmap, metric_file.str().c_str(), fullmap_length);
        }

        if (my_saved_node_count > 1) {
            PMPI_Barrier(MPI_COMM_WORLD);
            // share the full map length
            PMPI_Bcast(&fullmap_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (my_saved_node_id > 0) {
                fullmap = (char*) calloc(fullmap_length, sizeof(char));
            }

            // share the full map
            PMPI_Bcast(fullmap, fullmap_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        // read the reduced data
        if (my_saved_node_count > 1) {
            std::map<std::string,uint64_t> reduced_metric_map;
            std::string metric_line;
            std::stringstream metric_file(fullmap);
            std::string metric_name;
            int idx;
            // read the map from rank 0
            while (std::getline(metric_file, metric_line)) {
                istringstream ss(metric_line);
                ss >> idx >> metric_name;
                reduced_metric_map[metric_name] = idx;
            }
            // ...and distribute them back out
            write_metric_map(reduced_metric_map);
        }
     }

    std::string otf2_listener::write_my_threads(void) {
        stringstream metric_file;
        // first, output our number of threads.
        //metric_file << thread_instance::get_num_threads() << endl;
        metric_file << _event_threads.size() << endl;
        // then iterate over the threads and write them out.
        for (auto const i : _event_threads) {
            metric_file << i << "=" << _event_thread_names[i] << endl;
        }
        return metric_file.str();
    }

    void otf2_listener::reduce_threads(void) {
        std::string my_threads = write_my_threads();
        std::cout << my_threads.rdbuf() << std::endl;
        int length = my_threads.size();
        int full_length = 0;
        int max_length = 0;
        char * sbuf = nullptr;
        char * rbuf = nullptr;

        if (my_saved_node_count > 1) {
            // get the max length from all nodes
            PMPI_Allreduce(&length, &max_length, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD);
        } else {
            max_length = length;
        }
        // get the total length
        full_length = max_length * my_saved_node_count;

        // allocate space to store the strings on node 0
        if (my_saved_node_id == 0) {
            rbuf = (char*) calloc(full_length, sizeof(char));
        }
        // put the local data in a fixed length string
        sbuf = (char*) calloc(max_length, sizeof(char));
        strncpy(sbuf, my_threads.c_str(), max_length);

        if (my_saved_node_count > 1) {
            // gather the strings to node 0
            PMPI_Gather(sbuf, max_length, MPI_CHAR, rbuf,
                max_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            rbuf = sbuf;
        }

        int fullmap_length = 0;
        char * fullmap = nullptr;
        if (my_saved_node_id == 0) {
            // TODO!
        }
     }

#else

    /* When not using HPX or MPI, use the filesystem. Ick. */

    std::unique_ptr<std::tuple<std::map<int,int>,
                    std::map<int,std::string> > >
                    otf2_listener::reduce_node_properties(std::string&& str) {
        std::ofstream index_file(index_filename + to_string(my_saved_node_id));
        // write our info
        index_file << str.c_str();
        // close the file
        index_file.close();

        if (my_saved_node_id > 0) {
            return nullptr;
        }

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
            std::remove(full_index_filename.str().c_str());
        }
        return APEX_MAKE_UNIQUE<std::tuple<std::map<int,int>,
            std::map<int,string> > >(rank_pid_map, rank_hostname_map);
    }

    std::string otf2_listener::write_my_regions(void) {
        // create my lock file.
        ostringstream lock_filename;
        lock_filename << lock_filename_prefix << my_saved_node_id;
        ofstream lock_file(lock_filename.str(), ios::out | ios::trunc );
        lock_file << "lock" << endl;
        lock_file.close();
        lock_file.flush();
        // open my region file
        ostringstream region_filename;
        region_filename << region_filename_prefix << my_saved_node_id;
        ofstream region_file(region_filename.str(), ios::out | ios::trunc );
        // first, output our number of threads.
        //region_file << thread_instance::get_num_threads() << endl;
        region_file << _event_threads.size() << endl;
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
        return std::string();
    }

    int otf2_listener::reduce_regions(void) {
        write_my_regions();

        if (my_saved_node_id == 0) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock_filename_prefix << "all";
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
                region_line.erase(std::remove(region_line.begin(),
                    region_line.end(), '\n'), region_line.end());
                if (reduced_region_map.find(region_line) ==
                    reduced_region_map.end()) {
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
        region_filename << region_filename_prefix << "reduced." << my_saved_node_id;
        ofstream region_file(region_filename.str(), ios::out | ios::trunc );
        // copy the reduced map to a pair, so we can sort by value
        std::vector<std::pair<std::string, int>> pairs;
        for (auto const &i : reduced_region_map) {
            pairs.push_back(i);
        }
        sort(pairs.begin(), pairs.end(), string_sort_by_value());
        // iterate over the regions and write them out.
        for (auto const &i : pairs) {
            std::string name = i.first;
            uint64_t idx = i.second;
            region_file << idx << "\t" << name << endl;
        }
        // close the region file
        region_file.close();
        // delete the lock file, so everyone can read our data.
        int rc = std::remove(my_lock_filename.str().c_str());
        while (rc != 0) {
            rc = std::remove(my_lock_filename.str().c_str());
        }
        }

        // read the reduced data
        if (my_saved_node_count > 1) {
            struct stat buffer;
            std::map<std::string,uint64_t> reduced_region_map;
            // wait on the map file from rank 0 to exist
            ostringstream region_filename;
            region_filename << region_filename_prefix << "reduced." << 0;
            while (stat (region_filename.str().c_str(), &buffer) != 0) {}
            // wait for the lock file from rank 0 to NOT exist
            ostringstream lock_filename;
            lock_filename << lock_filename_prefix << "all";
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            std::string region_line;
            std::string region_name;
            std::ifstream region_file(region_filename.str());
            int idx;
            // read the map from rank 0
            while (std::getline(region_file, region_line)) {
                std::cout << region_line << std::endl;
                // find the first tab
                size_t index = region_line.find("\t");
                std::string tmp = region_line.substr(0,index);
                region_name = region_line.substr(index+1);
                idx = atoi(tmp.c_str());
                reduced_region_map[region_name] = idx;
                /* Debug output
                std::cout << my_saved_node_id << " PARSING"
                          << " idx: " << idx
                          << " id: " << region_name
                          << std::endl;
                 */
            }
            region_file.close();
            // ...and write the map to the local definitions
            write_region_map(reduced_region_map);
        }
        return my_saved_node_count;
    }

    std::string otf2_listener::write_my_metrics(void) {
        // create my lock file.
        ostringstream lock_filename;
        lock_filename << lock2_filename_prefix << my_saved_node_id;
        ofstream lock_file(lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // open my metric file
        ostringstream metric_filename;
        metric_filename << metric_filename_prefix << my_saved_node_id;
        ofstream metric_file(metric_filename.str(), ios::out | ios::trunc );
        // first, output our number of threads.
        //metric_file << thread_instance::get_num_threads() << endl;
        metric_file << _event_threads.size() << endl;
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
        return std::string();
    }

    void otf2_listener::reduce_metrics(void) {
        write_my_metrics();

        if (my_saved_node_id == 0) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock2_filename_prefix << "all";
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
            lock_filename << lock2_filename_prefix << i;
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
                metric_line.erase(std::remove(metric_line.begin(),
                    metric_line.end(), '\n'), metric_line.end());
                if (reduced_metric_map.find(metric_line) ==
                    reduced_metric_map.end()) {
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
        sort(pairs.begin(), pairs.end(), string_sort_by_value());
        // iterate over the metrics and write them out.
        for (auto const &i : pairs) {
            std::string name = i.first;
            uint64_t idx = i.second;
            metric_file << idx << "\t" << name << endl;
        }
        // close the metric file
        metric_file.close();
        // delete the lock file, so everyone can read our data.
        int rc = std::remove(my_lock_filename.str().c_str());
        while (rc != 0) {
            rc = std::remove(my_lock_filename.str().c_str());
        }
        }

        // read the reduced data
        if (my_saved_node_count > 1) {
            struct stat buffer;
            std::map<std::string,uint64_t> reduced_metric_map;
            // wait on the map file from rank 0 to exist
            ostringstream metric_filename;
            metric_filename << metric_filename_prefix << 0;
            while (stat (metric_filename.str().c_str(), &buffer) != 0) {}
            // wait for the lock file from rank 0 to NOT exist
            ostringstream lock_filename;
            lock_filename << lock2_filename_prefix << "all";
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            std::string metric_line;
            std::ifstream metric_file(metric_filename.str());
            std::string metric_name;
            int idx;
            // read the map from rank 0
            while (std::getline(metric_file, metric_line)) {
                istringstream ss(metric_line);
                ss >> idx >> metric_name;
                reduced_metric_map[metric_name] = idx;
            }
            metric_file.close();
            // ...and distribute them back out
            write_metric_map(reduced_metric_map);
        }
    }

    std::string otf2_listener::write_my_threads(void) {
        // create my lock file.
        ostringstream lock_filename;
        lock_filename << lock3_filename_prefix << my_saved_node_id;
        ofstream lock_file(lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // open my thread file
        ostringstream thread_filename;
        thread_filename << thread_filename_prefix << my_saved_node_id;
        ofstream thread_file(thread_filename.str(), ios::out | ios::trunc );
        // first, output our number of threads.
        //thread_file << thread_instance::get_num_threads() << endl;
        thread_file << _event_threads.size() << endl;
        // then iterate over the threads and write them out.
        for (auto const i : _event_threads) {
            thread_file << i << "=" << _event_thread_names[i] << endl;
        }
        // close the thread file
        thread_file.close();
        // delete the lock file, so rank 0 can read our data.
        std::remove(lock_filename.str().c_str());
        return std::string();
    }

    void otf2_listener::reduce_threads(void) {
        write_my_threads();

        if (my_saved_node_id == 0) {
        // create my lock file.
        ostringstream my_lock_filename;
        my_lock_filename << lock3_filename_prefix << "all";
        ofstream lock_file(my_lock_filename.str(), ios::out | ios::trunc );
        lock_file.close();
        // iterate over my thread map, and build a map of strings to ids
        // save my number of threads
        std::map<uint32_t, std::string> thread_name_map;
        for (auto const i : _event_threads) {
            thread_name_map[i] = _event_thread_names[i];
        }
        rank_thread_name_map[0] = std::move(thread_name_map);
        // iterate over the other ranks in the index file
        for (int i = 1 ; i < my_saved_node_count ; i++) {
            std::map<uint32_t, std::string> tmp_thread_name_map;
            struct stat buffer;
            // wait on the map file to exist
            ostringstream thread_filename;
            thread_filename << thread_filename_prefix << i;
            while (stat (thread_filename.str().c_str(), &buffer) != 0) {}
            // wait for the lock file to not exist
            ostringstream lock_filename;
            lock_filename << lock3_filename_prefix << i;
            while (stat (lock_filename.str().c_str(), &buffer) == 0) {}
            // get the number of threads from that rank
            std::string thread_line;
            std::ifstream thread_file(thread_filename.str());
            std::getline(thread_file, thread_line);
            // read the map from that rank
            while (std::getline(thread_file, thread_line)) {
                // trim the newline
                thread_line.erase(std::remove(thread_line.begin(),
                    thread_line.end(), '\n'), thread_line.end());
                uint32_t index = atol(strtok((char*)(thread_line.c_str()), "="));
                char * name = strtok(NULL, "=");
                tmp_thread_name_map.insert(std::pair<uint32_t,std::string>(index, std::string(name)));
            }
            // close the thread file
            thread_file.close();
            // remove that rank's map
            std::remove(thread_filename.str().c_str());
            rank_thread_name_map[i] = std::move(tmp_thread_name_map);
        }
        }
    }

#endif

    void otf2_listener::on_async_event(uint32_t device, uint32_t context,
        uint32_t stream, std::shared_ptr<profiler> &p) {
        // This could be a callback from a library before APEX is ready
        // Something like OpenMP or CUDA/CUPTI or...?
        if (!_initialized) return ;
        uint32_t tid{make_vtid(device, context, stream)};
        task_identifier * id = p->tt_ptr->get_task_id();
        uint64_t idx = get_region_index(id);
        //static map<uint32_t,std::string> last_p;
        if (last_ts.count(tid) == 0) {
            last_ts[tid] = 0ULL;
        }
        uint64_t last = last_ts[tid];
        // not likely, but just in case...
        if (_terminate) { return; }
        /* validate the time stamp.  CUPTI is notorious for giving out-of-order
         * events, so make sure this one isn't before the previous. */
        uint64_t stamp = 0L;
        stamp = p->get_start_ns() - globalOffset;
        if(stamp < last) {
            dropped++;
            /*
            std::cerr << "APEX: Warning - Events delivered out of order on Device "
                      << device << ", Context " << context << ", Stream " << stream
                      << ".\nIgnoring event " << p->tt_ptr->task_id->get_name()
                      << " with timestamp of " << stamp << " after last event "
                      << "with timestamp of " << last << std::endl;
                      */
            return;
        }
        // don't close the archive on us!
        read_lock_type lock(_archive_mutex);
        // before we process the event, make sure the event write is open
        OTF2_EvtWriter* local_evt_writer = vthread_evt_writer_map[tid];
        if (local_evt_writer != nullptr) {
            // create an attribute list
            OTF2_AttributeList * al = OTF2_AttributeList_New();
            // create an attribute
            OTF2_AttributeList_AddUint64( al, 0, p->tt_ptr->guid );
            OTF2_AttributeList_AddUint64( al, 1, p->tt_ptr->parent_guid );
            OTF2_EC(OTF2_EvtWriter_Enter( local_evt_writer, al,
                stamp, idx /* region */ ));
            stamp = p->get_stop_ns() - globalOffset;
            OTF2_EC(OTF2_EvtWriter_Leave( local_evt_writer, al,
                stamp, idx /* region */ ));
            last_ts[tid] = stamp;
            //last_p[tid] = std::string(p->tt_ptr->task_id->get_name());
            // delete the attribute list
            OTF2_AttributeList_Delete(al);
        }
        return;

    }

}
