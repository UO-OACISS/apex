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

namespace apex {
namespace otf2 {

int64_t getClockOffset(void) {
    int64_t offset{0};
    uint64_t offset_time{0};
    int rank{0};
    int size{0};
    MPI_Status status;

    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    // synchronize all ranks
    PMPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        /* If rank 0, iterate over the other ranks and synchronize */
        for (int index = 1 ; index < size ; index++) {
            /* Measure how long it takes to send/receive with the worker */
            uint64_t latency;
            // take a timestamp
            uint64_t before = apex::otf2_listener::get_time();
            // send an empty message
            PMPI_Send(NULL, 0, MPI_INT, index, 1, MPI_COMM_WORLD);
            // receive an empty message
            PMPI_Recv(NULL, 0, MPI_INT, index, 2, MPI_COMM_WORLD, &status);
            // take a timestamp
            uint64_t after = apex::otf2_listener::get_time();
            // the latency is half of the elapsed time
            latency = (after - before) / 2;
            /* Set the reference time stamp for this worker to the rank 0
               "before" time plus the latency.  That should match when
               the worker took their time stamp, if the two are well synced. */
            uint64_t ref_time = before + latency;
            PMPI_Send(&ref_time, 1, MPI_UNSIGNED_LONG_LONG, index, 1, MPI_COMM_WORLD);
            offset = 0;
        }
    } else {
        /* Measure how long it takes to send/receive with the main rank */
        PMPI_Recv(NULL, 0, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        // take a timestamp now!
        uint64_t mytime = apex::otf2_listener::get_time();
        PMPI_Send(NULL, 0, MPI_INT, 0, 2, MPI_COMM_WORLD);
        // get the reference time from rank 0
        uint64_t ref_time;
        PMPI_Recv(NULL, &ref_time, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD, &status);
        // our offset is the reference time minus our timestamp between messages.
        offset = ref_time - mytime;
    }
    // synchronize all ranks again
    PMPI_Barrier(MPI_COMM_WORLD);

    return offset;
}

} // namespace otf2
} // namespace apex

#endif