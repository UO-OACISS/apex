/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "profile_reducer.hpp"
#include "apex.hpp"
#include "string.h"
#include <vector>
#include <iostream>
#include <map>
#include <limits>
#include <vector>
#include <inttypes.h>
#include "csv_parser.h"
#include "tree.h"
#include "threadpool.h"
#include <chrono>
using namespace std::chrono;


/* 11 values per timer/counter by default
 * 4 values related to memory allocation tracking
 * 8 values (up to) when PAPI enabled */
constexpr size_t num_fields{23};

#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
#include "mpi.h"
#endif

// Macro to check MPI calls status
#define MPI_CALL(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char errstr[MPI_MAX_ERROR_STRING]; \
            int errlen; \
            MPI_Error_string(err, errstr, &errlen); \
            fprintf(stderr, "%s\n", errstr); \
            MPI_Abort(MPI_COMM_WORLD, 999); \
        } \
    } while (0)


/* Main routine to reduce profiles across all ranks for MPI applications */

namespace apex {

std::map<std::string, apex_profile*> reduce_profiles_for_screen() {
    int commrank = 0;
    int commsize = 1;
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    int mpi_initialized = 0;
    MPI_CALL(MPI_Initialized( &mpi_initialized ));
    if (mpi_initialized) {
        MPI_CALL(PMPI_Comm_rank(MPI_COMM_WORLD, &commrank));
        MPI_CALL(PMPI_Comm_size(MPI_COMM_WORLD, &commsize));
    }
#endif

    std::map<std::string, apex_profile*> all_profiles;
    /* Get a list of all profile names */
    std::vector<task_identifier>& tids = get_available_profiles();
    /* Build a map of names to IDs.  I know, that sounds weird but hear me out.
     * Timers that are identified by address have to be looked up by address,
     * not by name.  So we map the names to ids, then use the ids to look them up. */
    std::map<std::string, task_identifier> tid_map;

    /* check for no data */
    if (tids.size() < 1) { return (all_profiles); }

    /* Build a set of the profiles of interest */
    std::set<std::string> all_names;
    for (auto tid : tids) {
        std::string tmp{tid.get_name()};
        // skip APEX MAIN, it's bogus anyway
        if (tmp.compare(APEX_MAIN_STR) == 0) { continue; }
        //DEBUG_PRINT("%d Inserting: %s\n", commrank, tmp.c_str());
        all_names.insert(tmp);
        tid_map.insert(std::pair<std::string, task_identifier>(tmp, tid));
    }
    size_t length[2];
    size_t max_length[2];

    /* the number of profiles */
    length[0] = all_names.size();

    /* get the max string length for all profiles */
    length[1] = 0;
    for (auto name : all_names) {
        size_t len = name.size();
        if (len > length[1]) {
            length[1] = len;
        }
    }
    // add a "spacer" between the longest string and its neighbor
    length[1] = length[1] + 1;

    /* AllReduce all profile name counts */
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Allreduce(&length, &max_length, 2,
            MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
    } else {
#else
    if (true) {
#endif
        max_length[0] = length[0];
        max_length[1] = length[1];
    }

    /* Allocate space */
    //DEBUG_PRINT("%d:%d Found %" PRIu64 " strings of max length %" PRIu64 "\n", commrank, commsize, max_length[0], max_length[1]);
    size_t sbuf_length = max_length[0] * max_length[1];
    size_t rbuf_length = max_length[0] * max_length[1] * commsize;
    char * sbuf = (char*)calloc(sbuf_length, sizeof(char));
    char * rbuf = (char*)calloc(rbuf_length, sizeof(char));

    /* Allgather all profile names */
    char * ptr = sbuf;
    for (auto name : all_names) {
        strncpy(ptr, name.c_str(), max_length[1]);
        ptr = ptr + max_length[1];
    }
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Allgather(sbuf, sbuf_length, MPI_CHAR,
            rbuf, sbuf_length, MPI_CHAR, MPI_COMM_WORLD));
    } else {
#else
    if (true) {
#endif
        free(rbuf);
        rbuf = sbuf;
    }

    /* Iterate over gathered names and add new ones to the set */
    for (char * ptr = rbuf ; ptr < rbuf + rbuf_length ; ptr += max_length[1]) {
        std::string tmp{ptr};
        if (all_names.count(tmp) == 0) {
            all_names.insert(tmp);
            //DEBUG_PRINT("%d Inserting: %s\n", commrank, tmp.c_str());
        }
    }
    // the set is already sorted?...
    //sort(all_names.begin(), all_names.end());

    // There are 8 "values" and 8 possible papi counters
    sbuf_length = all_names.size() * num_fields;
    rbuf_length = all_names.size() * num_fields * commsize;
    //DEBUG_PRINT("%d Sending %" PRIu64 " bytes\n", commrank, sbuf_length * sizeof(double));
    double * s_pdata = (double*)calloc(sbuf_length, sizeof(double));
    double * r_pdata = nullptr;
    if (commrank == 0) {
        //DEBUG_PRINT("%d Receiving %" PRIu64 " bytes\n", commrank, rbuf_length * sizeof(double));
        r_pdata = (double*)calloc(rbuf_length, sizeof(double));
    }

    /* Build array of data */
    double * dptr = s_pdata;
    for (auto name : all_names) {
        auto tid = tid_map.find(name);
        if (tid != tid_map.end()) {
        auto p = get_profile(tid->second);
        if (p != nullptr) {
            int i{0};
            dptr[i++] = p->calls == 0.0 ? 1 : p->calls;
            dptr[i++] = p->stops == 0.0 ? 1 : p->stops;
            dptr[i++] = p->accumulated;
            dptr[i++] = p->inclusive_accumulated;
            dptr[i++] = p->sum_squares;
            dptr[i++] = p->minimum;
            dptr[i++] = p->maximum;
            dptr[i++] = p->times_reset;
            dptr[i++] = (double)p->type;
            dptr[i++] = p->num_threads;
            dptr[i++] = (p->throttled ? 1.0 : 0.0);
            dptr[i++] = p->allocations;
            dptr[i++] = p->frees;
            dptr[i++] = p->bytes_allocated;
            dptr[i++] = p->bytes_freed;
            if (p->type == APEX_TIMER) {
                dptr[i++] = p->papi_metrics[0];
                dptr[i++] = p->papi_metrics[1];
                dptr[i++] = p->papi_metrics[2];
                dptr[i++] = p->papi_metrics[3];
                dptr[i++] = p->papi_metrics[4];
                dptr[i++] = p->papi_metrics[5];
                dptr[i++] = p->papi_metrics[6];
                dptr[i++] = p->papi_metrics[7];
            }
        }
        }
        dptr = &(dptr[num_fields]);
    }

    /* Reduce the data */
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Gather(s_pdata, sbuf_length, MPI_DOUBLE,
            r_pdata, sbuf_length, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    } else {
#else
    if (true) {
#endif
        free(r_pdata);
        r_pdata = s_pdata;
    }

    /* We're done with everyone but rank 0 */
    if (commrank == 0) {

    /* Iterate over results, and build up the profile on rank 0 */
    dptr = r_pdata;
    for (int i = 0 ; i < commsize ; i++) {
        for (auto name : all_names) {
            auto next = all_profiles.find(name);
            apex_profile* p;
            if (next == all_profiles.end()) {
                p = (apex_profile*)calloc(1, sizeof(apex_profile));
                p->type = (apex_profile_type)(dptr[7]);
                // set the minimum to something rediculous
                p->minimum = std::numeric_limits<double>::max();
                all_profiles.insert(std::pair<std::string, apex_profile*>(name, p));
            } else {
                p = next->second;
            }
            int index{0};
            p->calls += dptr[index++];
            p->stops += dptr[index++];
            p->accumulated += dptr[index++];
            p->inclusive_accumulated += dptr[index++];
            p->sum_squares += dptr[index++];
            p->minimum = dptr[index] < p->minimum ? dptr[index] : p->minimum;
            index++;
            p->maximum = dptr[index] > p->maximum ? dptr[index] : p->maximum;
            index++;
            p->times_reset += dptr[index++];
            p->type = (apex_profile_type)(dptr[index++]);
            p->num_threads = dptr[index] > p->num_threads ? dptr[index] : p->num_threads;
            index++;
            p->throttled = (p->throttled || (dptr[index++] > 0.0)) ? true : false;
            p->allocations = dptr[index++];
            p->frees = dptr[index++];
            p->bytes_allocated = dptr[index++];
            p->bytes_freed = dptr[index++];
            if (p->type == APEX_TIMER) {
                p->papi_metrics[0] += dptr[index++];
                p->papi_metrics[1] += dptr[index++];
                p->papi_metrics[2] += dptr[index++];
                p->papi_metrics[3] += dptr[index++];
                p->papi_metrics[4] += dptr[index++];
                p->papi_metrics[5] += dptr[index++];
                p->papi_metrics[6] += dptr[index++];
                p->papi_metrics[7] += dptr[index++];
            }
            dptr = &(dptr[num_fields]);
        }
    }

    }
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Barrier(MPI_COMM_WORLD));
    }
#endif
    return (all_profiles);
}

    void reduce_profiles(std::stringstream& header, std::stringstream& csv_output, std::string filename, bool flat) {
        int commrank = 0;
        int commsize = 1;
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
        int mpi_initialized = 0;
        MPI_CALL(MPI_Initialized( &mpi_initialized ));
        if (mpi_initialized) {
            MPI_CALL(PMPI_Comm_rank(MPI_COMM_WORLD, &commrank));
            MPI_CALL(PMPI_Comm_size(MPI_COMM_WORLD, &commsize));
        }
#endif
        // if nothing to reduce, just write the data.
        if (commsize == 1) {
            std::ofstream csvfile;
            std::stringstream csvname;
            csvname << apex_options::output_file_path();
            csvname << filesystem_separator() << filename;
            std::cout << "Writing: " << csvname.str() << std::endl;
            csvfile.open(csvname.str(), std::ios::out);
            csvfile << csv_output.str();
            csvfile.close();
            return;
        }

        size_t length{csv_output.str().size()};
        size_t max_length{length};
        // get the longest string from all ranks
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
        if (mpi_initialized && commsize > 1) {
            MPI_CALL(PMPI_Allreduce(&length, &max_length, 1,
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
        }
        // so we don't have to specially handle the first string which will append
        // the second string without a null character (zero).
        max_length = max_length + 1;
#endif
        // allocate the send buffer
        char * sbuf = (char*)calloc(max_length, sizeof(char));
        // copy into the send buffer
        strncpy(sbuf, csv_output.str().c_str(), length);
        // allocate the memory to hold all output
        char * rbuf = nullptr;
        if (commrank == 0) {
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
            rbuf = (char*)calloc(max_length * commsize, sizeof(char));
#else
            rbuf = sbuf;
#endif
        }

#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
        MPI_Gather(sbuf, max_length, MPI_CHAR, rbuf, max_length, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

        /* OK, we have one big blob of character data. Split it into ranks... */
        /*
        char * index = rbuf;
        for (auto i = 0 ; i < commsize ; i++) {
            index = rbuf+(i*max_length);
            std::string tmpstr{index};

        }
        */

        /* Write to /dev/shm/apex.tmpfile.csv */
        if (commrank == 0 && flat) {
            std::ofstream csvfile;
            std::stringstream csvname;
            csvname << apex_options::output_file_path();
            csvname << filesystem_separator() << filename;
            std::cout << "Writing: " << csvname.str();
            csvfile.open(csvname.str(), std::ios::out);
            csvfile << header.rdbuf();
            char * index = rbuf;
            for (auto i = 0 ; i < commsize ; i++) {
                index = rbuf+(i*max_length);
                std::string tmpstr{index};
                csvfile << tmpstr;
                csvfile.flush();
            }
            csvfile.close();
            std::cout << "...done." << std::endl;
            free(sbuf);
            free(rbuf);
        } else if (commrank == 0 && !flat) {
            std::vector<std::vector<std::vector<std::string>>*> ranks;
            std::vector<std::vector<std::string>> *rows = new std::vector<std::vector<std::string>>{};
            treemerge::ThreadPool pool{};
            pool.Start();
            treemerge::node * root{nullptr};
            std::cout << "Merging common tree for all ranks... ";
            auto start = high_resolution_clock::now();

            char * index = rbuf;
            for (auto i = 0 ; i < commsize ; i++) {
                index = rbuf+(i*max_length);
                std::string tmpstr{index};
                std::istringstream iss{tmpstr};
                while (iss.good()) {
                    std::vector<std::string> row = treemerge::csv_read_row(iss, ',');
                    rows->push_back(row);
                }
                //std::cout << "Read " << rows->size() << " rows for rank " << ranks.size() << std::endl;
                ranks.push_back(rows);
                rows = new std::vector<std::vector<std::string>>{};
                // if this is the root node, synchronously start the common tree
                if (i == 0) {
                    root = treemerge::node::buildTree(*(ranks[i]), root);
                } else {
                    //std::cout << "Queueing " << i << std::endl;
                    pool.QueueJob([&ranks,i,&root]() {
                        treemerge::node::buildTree(*(ranks[i]), root);
                    });
                }
            }
            free(sbuf);
            free(rbuf);
            while(pool.busy()) {} // wait for jobs to complete
            pool.Stop();
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "done in " << duration.count() << " ms with "
                << pool.getNthreads() << " threads." << std::endl;
            std::cout << "Common task tree has " << root->getSize()
                << " nodes." << std::endl;
            delete(root);

            /* now write the common tree! */
            std::ofstream csvfile;
            std::stringstream csvname;
            csvname << apex_options::output_file_path();
            csvname << filesystem_separator() << filename;
            std::cout << "Writing: " << csvname.str() << std::endl;
            std::istringstream iss{header.str()};
            std::vector<std::string> header2 = treemerge::csv_read_row(iss, ',');
            treemerge::csv_write(header2, ranks, csvname.str());
            for (auto r : ranks) {
                delete(r);
            }
        } else {
            free(sbuf);
            free(rbuf);
        }
    }

    void reduce_flat_profiles(int node_id, int num_papi_counters,
        std::vector<std::string> metric_names, profiler_listener* listener) {
#ifndef APEX_HAVE_PAPI // prevent compiler warnings
        APEX_UNUSED(num_papi_counters);
        APEX_UNUSED(metric_names);
#endif
        std::stringstream header;
        if (node_id == 0) {
            header << "\"rank\",\"name\",\"type\",\"num samples/calls\",\"yields\",\"minimum\",\"mean\","
                << "\"maximum\",\"stddev\",\"total\",\"inclusive (ns)\",\"num threads\",\"total per thread\"";
#if APEX_HAVE_PAPI
            for (int i = 0 ; i < num_papi_counters ; i++) {
                header << ",\"" << metric_names[i] << "\"";
            }
#endif
            if (apex_options::track_cpu_memory() || apex_options::track_gpu_memory()) {
                header << ",\"allocations\", \"bytes allocated\", \"frees\", \"bytes freed\"";
            }
            header << std::endl;
        }
        std::stringstream csv_output;

        /* Get a list of all profile names */
        std::vector<task_identifier>& tids = get_available_profiles();
        for (auto tid : tids) {
            std::string name{tid.get_name()};
            auto p = listener->get_profile(tid);
            csv_output << node_id << ",\"" << name << "\",";
            if (p->get_type() == APEX_TIMER) {
                csv_output << "\"timer\",";
            } else {
                csv_output << "\"counter\",";
            }
            csv_output << llround(p->get_calls()) << ",";
            csv_output << llround(p->get_stops() - p->get_calls()) << ",";
            // add all the extra columns for counter and timer data
            csv_output << std::llround(p->get_minimum()) << ",";
            csv_output << std::llround(p->get_mean()) << ",";
            csv_output << std::llround(p->get_maximum()) << ",";
            csv_output << std::llround(p->get_stddev()) << ",";
            csv_output << std::llround(p->get_accumulated()) << ",";
            if (p->get_type() == APEX_TIMER) {
                csv_output << std::llround(p->get_inclusive_accumulated()) << ",";
            } else {
                csv_output << std::llround(0.0) << ",";
            }
            csv_output << std::llround(p->get_num_threads()) << ",";
            csv_output << std::llround(p->get_accumulated()/p->get_num_threads());
#if APEX_HAVE_PAPI
            for (int i = 0 ; i < num_papi_counters ; i++) {
                csv_output << "," << std::llround(p->get_papi_metrics()[i]);
            }
#endif
            if (apex_options::track_cpu_memory() || apex_options::track_gpu_memory()) {
                csv_output << "," << p->get_allocations();
                csv_output << "," << p->get_bytes_allocated();
                csv_output << "," << p->get_frees();
                csv_output << "," << p->get_bytes_freed();
            }
            csv_output << std::endl;
        }
        reduce_profiles(header, csv_output, "apex_profiles.csv", true);
    }

} // namespace

