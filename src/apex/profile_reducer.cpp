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
#include <inttypes.h>

#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
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

std::map<std::string, apex_profile*> reduce_profiles() {
    int commrank = 0;
    int commsize = 1;
    int mpi_initialized = 0;
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
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
        if (tmp.compare("APEX MAIN") == 0) { continue; }
        //DEBUG_PRINT("%d Inserting: %s\n", commrank, tmp.c_str());
        all_names.insert(tmp);
        tid_map.insert(std::pair(tmp, tid));
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
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
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
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
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

    // There are 6 "values" and 8 possible papi counters
    sbuf_length = all_names.size() * 15;
    rbuf_length = all_names.size() * 15 * commsize;
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
            dptr[0] = p->calls == 0.0 ? 1 : p->calls;
            dptr[1] = p->accumulated;
            dptr[2] = p->sum_squares;
            dptr[3] = p->minimum;
            dptr[4] = p->maximum;
            dptr[5] = p->times_reset;
            dptr[6] = (double)p->type;
            if (p->type == APEX_TIMER) {
                dptr[7] = p->papi_metrics[0];
                dptr[8] = p->papi_metrics[1];
                dptr[9] = p->papi_metrics[2];
                dptr[10] = p->papi_metrics[3];
                dptr[11] = p->papi_metrics[4];
                dptr[12] = p->papi_metrics[5];
                dptr[13] = p->papi_metrics[6];
                dptr[14] = p->papi_metrics[7];
            }
        }
        }
        dptr = &(dptr[15]);
    }

    /* Reduce the data */
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
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
                p->type = (apex_profile_type)(dptr[6]);
                // set the minimum to something rediculous
                p->minimum = std::numeric_limits<double>::max();
                all_profiles.insert(std::pair(name, p));
            } else {
                p = next->second;
            }
            p->calls += dptr[0];
            p->accumulated += dptr[1];
            p->sum_squares += dptr[2];
            p->minimum = dptr[3] < p->minimum ? dptr[3] : p->minimum;
            p->maximum = dptr[4] > p->maximum ? dptr[4] : p->maximum;
            p->times_reset += dptr[5];
            if (p->type == APEX_TIMER) {
                p->papi_metrics[0] += dptr[7];
                p->papi_metrics[1] += dptr[8];
                p->papi_metrics[2] += dptr[9];
                p->papi_metrics[3] += dptr[10];
                p->papi_metrics[4] += dptr[11];
                p->papi_metrics[5] += dptr[12];
                p->papi_metrics[6] += dptr[13];
                p->papi_metrics[7] += dptr[14];
            }
            dptr = &(dptr[15]);
        }
    }

    }
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Barrier(MPI_COMM_WORLD));
    }
#endif
    return (all_profiles);
}

}
