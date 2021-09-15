/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This file contains MPI implementations of communication necessary to support
   OTF2 tracing.  For example, event unification and clock Synchronization.
*/

// only compile this file if we have MPI support (but not HPX!)
#if !defined(HPX_HAVE_NETWORKING) && defined(APEX_HAVE_MPI)

#include "otf2_listener.hpp"
#include "mpi.h"
#include <inttypes.h>


namespace apex {

int64_t otf2_listener::synchronizeClocks(void) {
    int64_t offset{0};
    int rank{getCommRank()};
    int size{getCommSize()};
    MPI_Status status;
    const int attempts{10};

    /* Check to make sure we're actually in an MPI application */
    int initialized;
    PMPI_Initialized(&initialized);
    if (!initialized) {
        return offset;
    }

    // synchronize all ranks
    PMPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        /* If rank 0, iterate over the other ranks and synchronize */
        for (int index = 1 ; index < size ; index++) {
            /* Measure how long it takes to send/receive with the worker */
            uint64_t before[attempts];
            uint64_t after[attempts];
            // take a timestamp
            for (int i = 0 ; i < attempts ; i++) {
                before[i] = otf2_listener::get_time();
                // send an empty message
                PMPI_Send(NULL, 0, MPI_INT, index, 1, MPI_COMM_WORLD);
                // receive an empty message
                PMPI_Recv(NULL, 0, MPI_INT, index, 2, MPI_COMM_WORLD, &status);
                // take a timestamp
                after[i] = otf2_listener::get_time();
            }
            uint64_t latency = after[0] - before[0];
            int my_min{0};
            for (int i = 1 ; i < attempts ; i++) {
                uint64_t next = after[i] - before[i];
                if (next < latency) {
                    latency = next;
                    my_min = i;
                }
            }
            // the latency is half of the elapsed time
            latency = (after[my_min] - before[my_min]) / 2;
            /* Set the reference time stamp for this worker to the rank 0
               "before" time plus the latency.  That should match when
               the worker took their time stamp, if the two are well synced. */
            uint64_t ref_time[2];
            ref_time[0] = after[my_min] - latency;
            ref_time[1] = my_min;
            PMPI_Send(ref_time, 2, MPI_UNSIGNED_LONG_LONG, index, 1, MPI_COMM_WORLD);
            offset = 0;
            //printf("0->%d: Before: %" PRIu64 "   After: %" PRIu64 " Latency: %" PRIu64 "\n", index, before[my_min], after[my_min], latency);
        }
    } else {
        uint64_t mytime[attempts];
        for (int i = 0 ; i < attempts ; i++) {
            /* Measure how long it takes to send/receive with the main rank */
            PMPI_Recv(NULL, 0, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            // take a timestamp now!
            mytime[i] = otf2_listener::get_time();
            PMPI_Send(NULL, 0, MPI_INT, 0, 2, MPI_COMM_WORLD);
        }
        // get the reference time from rank 0
        uint64_t ref_time[2];
        PMPI_Recv(ref_time, 2, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD, &status);
        // our offset is the reference time minus our timestamp between messages.
        offset = ref_time[0] - mytime[ref_time[1]];
        //printf("   %d: mytime: %" PRIu64 " reftime: %" PRIu64 "  offset: %" PRId64" \n", rank, mytime[ref_time[1]], ref_time[0], offset);
    }
    // synchronize all ranks again
    PMPI_Barrier(MPI_COMM_WORLD);

    return offset;
}

int otf2_listener::getCommRank() {
    static int rank{-1};
    if (rank == -1) {
        int initialized;
        PMPI_Initialized(&initialized);
        if (initialized) {
            PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        } else {
            rank = 0;
        }
    }
    return rank;
}

int otf2_listener::getCommSize() {
    static int size{-1};
    if (size == -1) {
        int initialized;
        PMPI_Initialized(&initialized);
        if (initialized) {
            PMPI_Comm_size(MPI_COMM_WORLD, &size);
        } else {
            size = 1;
        }
    }
    return size;
}

} // namespace apex

#endif

