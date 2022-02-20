/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#include <hpx/modules/threading_base.hpp>
#endif

#include "apex_api.hpp"
#if defined(APEX_HAVE_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
#include "mpi.h"
#ifndef MPI_Request
typedef int MPI_Request;
#endif
#endif

using namespace apex;

/* Implementation of the C API */

extern "C" {

/* When running with MPI and OTF (or other event unification at the end of
 * execution) we need to finalize APEX before MPI_Finalize() is called, so
 * that we can use MPI for the wrap-up.  We can override the weak MPI
 * implementation of Finalize, and do what we need to. */
#if defined(APEX_HAVE_MPI) && !defined(HPX_HAVE_NETWORKING)
    int MPI_Finalize(void) {
        finalize();
        int retval = PMPI_Finalize();
        cleanup();
        return retval;
    }
#endif
#if defined(APEX_HAVE_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    /* There are also a handful of interesting function calls that HPX uses
       that we should measure when requested */
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
        int tag, MPI_Comm comm, MPI_Request *request) {
        /* Get the byte count */
        int typesize = 0;
        PMPI_Type_size( datatype, &typesize );
        double bytes = (double)(typesize) * (double)(count);
        /* start the timer */
        auto p = start(__func__);
        /* sample the bytes */
        sample_value("MPI_Send : Bytes", bytes);
        int retval = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
        stop(p);
        /* record the bandwidth */
        sample_value("MPI_Send : BW (Bytes/second)", bytes/p->elapsed_seconds());
        return retval;
    }
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request *request) {
        auto p = start(__func__);
        int retval = PMPI_Irecv(buf, count, datatype, source, tag, comm,
            request);
        stop(p);
        return retval;
    }
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
        int tag, MPI_Comm comm){
        /* Get the byte count */
        int typesize = 0;
        PMPI_Type_size( datatype, &typesize );
        double bytes = (double)(typesize) * (double)(count);
        /* start the timer */
        auto p = start(__func__);
        /* sample the bytes */
        sample_value("MPI_Send : Bytes", bytes);
        int retval = PMPI_Send(buf, count, datatype, dest, tag, comm);
        stop(p);
        /* record the bandwidth */
        sample_value("MPI_Send : BW (Bytes/second)", bytes/p->elapsed_seconds());
        return retval;
    }
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Status *status){
        auto p = start(__func__);
        int retval = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
        stop(p);
        return retval;
    }
    int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf,
            recvcount, recvtype, root, comm);
        stop(p);
        return retval;
    }
    /* There are a handful of interesting Collectives! */
    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        stop(p);
        return retval;
    }
    int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, int root, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        stop(p);
        return retval;
    }
    int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
        MPI_Comm comm ) {
        auto p = start(__func__);
        int retval = PMPI_Bcast(buffer, count, datatype, root, comm );
        stop(p);
        return retval;
    }
    int MPI_Startall(int count, MPI_Request array_of_requests[]) {
        auto p = start(__func__);
        int retval = PMPI_Startall(count, array_of_requests);
        stop(p);
        return retval;
    }
    int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        stop(p);
        return retval;
    }
    int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        stop(p);
        return retval;
    }
    int MPI_Allgatherv(const void* buffer_send, int count_send, MPI_Datatype datatype_send,
        void* buffer_recv, const int* counts_recv, const int* displacements,
        MPI_Datatype datatype_recv, MPI_Comm communicator) {
        auto p = start(__func__);
        int retval = PMPI_Allgatherv(buffer_send, count_send, datatype_send,
            buffer_recv, counts_recv, displacements, datatype_recv, communicator);
        stop(p);
        return retval;
    }
    int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs,
        MPI_Datatype recvtype, int root, MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
        stop(p);
        return retval;
    }
    int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int source, int recvtag, MPI_Comm comm, MPI_Status * status) {
        auto p = start(__func__);
        int retval = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
                 source, recvtag, comm, status);
        stop(p);
        return retval;
    }
    /* There are a handful of interesting Synchronization functions */
    int MPI_Waitall(int count, MPI_Request array_of_requests[],
        MPI_Status array_of_statuses[]) {
        auto p = start(__func__);
        int retval = PMPI_Waitall(count, array_of_requests, array_of_statuses);
        stop(p);
        return retval;
    }
    int MPI_Wait(MPI_Request *request, MPI_Status *status) {
        auto p = start(__func__);
        int retval = PMPI_Wait(request, status);
        stop(p);
        return retval;
    }
    int MPI_Barrier(MPI_Comm comm) {
        auto p = start(__func__);
        int retval = PMPI_Barrier(comm);
        stop(p);
        return retval;
    }
    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
        auto p = start(__func__);
        int retval = PMPI_Test(request, flag, status);
        stop(p);
        return retval;
    }
    int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx,
        int *flag, MPI_Status *status) {
        auto p = start(__func__);
        int retval = PMPI_Testany(count, array_of_requests, indx, flag, status);
        stop(p);
        return retval;
    }
#endif

} // extern "C"


