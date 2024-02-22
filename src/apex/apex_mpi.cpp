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
#include "memory_wrapper.hpp"
#include "apex_error_handling.hpp"
#include "proc_read.h"
#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
#include "mpi.h"
#endif

#define MPI_START_TIMER auto p = apex::new_task(__APEX_FUNCTION__); apex::start(p);
#define MPI_STOP_TIMER apex::stop(p);

/* Implementation of the C API */

extern "C" {

/* Helpers to convert predefined types from Fortran to C */

/* Don't do this for HPX */
#ifndef HPX_HAVE_NETWORKING
extern void apex_mpi_fortran_init_predefined_constants_();
#endif

bool do_setup() {
#ifndef HPX_HAVE_NETWORKING
    apex_mpi_fortran_init_predefined_constants_();
#endif
    return true;
}
#define APEX_CREATE_CONVERTERRS \
static bool converters_init = do_setup(); \
APEX_UNUSED(converters_init);

static void ** mpi_predef_in_place(void)
{
  static void * in_place_ptr = NULL;
  return &in_place_ptr;
}

static void ** mpi_predef_bottom(void)
{
  static void * mpi_bottom_ptr = NULL;
  return &mpi_bottom_ptr;
}

static void ** mpi_predef_status_ignore(void)
{
  static void * mpi_status_ignore_ptr = NULL;
  return &mpi_status_ignore_ptr;
}

static void ** mpi_predef_statuses_ignore(void)
{
  static void * mpi_statuses_ignore_ptr = NULL;
  return &mpi_statuses_ignore_ptr;
}

static void ** mpi_predef_unweighted(void)
{
  static void * mpi_unweighted_ptr = NULL;
  return &mpi_unweighted_ptr;
}

void apex_mpi_predef_init_in_place(void * in_place) {
  *(mpi_predef_in_place()) = in_place;
  //std::cout << "Fortran MPI_IN_PLACE: " << in_place << ", C MPI_IN_PLACE: " << MPI_IN_PLACE << std::endl;
}
void apex_mpi_predef_init_in_place_(void * in_place) {
  apex_mpi_predef_init_in_place(in_place);
  //std::cout << "Fortran MPI_IN_PLACE: " << in_place << ", C MPI_IN_PLACE: " << MPI_IN_PLACE << std::endl;
}
void apex_mpi_predef_init_bottom(void * bottom) {
  *(mpi_predef_bottom()) = bottom;
  //std::cout << "Fortran MPI_BOTTOM: " << bottom << ", C MPI_BOTTOM: " << MPI_BOTTOM << std::endl;
}
void apex_mpi_predef_init_bottom_(void * bottom) {
  apex_mpi_predef_init_bottom(bottom);
  //std::cout << "Fortran MPI_BOTTOM: " << bottom << ", C MPI_BOTTOM: " << MPI_BOTTOM << std::endl;
}
void apex_mpi_predef_init_status_ignore(void * status_ignore) {
  *(mpi_predef_status_ignore()) = status_ignore;
  //std::cout << "Fortran MPI_STATUS_IGNORE: " << status_ignore << ", C MPI_STATUS_IGNORE: " << MPI_STATUS_IGNORE << std::endl;
}
void apex_mpi_predef_init_status_ignore_(void * status_ignore) {
  *(mpi_predef_status_ignore()) = status_ignore;
  //std::cout << "Fortran MPI_STATUS_IGNORE: " << status_ignore << ", C MPI_STATUS_IGNORE: " << MPI_STATUS_IGNORE << std::endl;
}
void apex_mpi_predef_init_statuses_ignore(void * statuses_ignore) {
  *(mpi_predef_statuses_ignore()) = statuses_ignore;
  //std::cout << "Fortran MPI_STATUSES_IGNORE: " << statuses_ignore << ", C MPI_STATUSES_IGNORE: " << MPI_STATUSES_IGNORE << std::endl;
}
void apex_mpi_predef_init_statuses_ignore_(void * statuses_ignore) {
  *(mpi_predef_statuses_ignore()) = statuses_ignore;
  //std::cout << "Fortran MPI_STATUSES_IGNORE: " << statuses_ignore << ", C MPI_STATUSES_IGNORE: " << MPI_STATUSES_IGNORE << std::endl;
}
void apex_mpi_predef_init_unweighted(void * unweighted) {
  *(mpi_predef_unweighted()) = unweighted;
}
void apex_mpi_predef_init_unweighted_(void * unweighted) {
  *(mpi_predef_unweighted()) = unweighted;
}


int MPI_Abort( MPI_Comm comm, int errorcode ) {
    apex_print_backtrace();
    return PMPI_Abort(comm, errorcode);
}
void  mpi_abort_( MPI_Fint *comm, MPI_Fint *errorcode, MPI_Fint *ierr) {
  *ierr = MPI_Abort( MPI_Comm_f2c(*comm), *errorcode );
}

bool amIroot(MPI_Comm comm, int root) {
    static std::map<MPI_Comm, int> theMap;
    if (theMap.count(comm) == 0) {
        int rank{0};
        PMPI_Comm_rank(comm, &rank);
        theMap.insert(std::pair<MPI_Comm, int>(comm, rank));
    }
    return (theMap[comm] == root);
}

/* When running with MPI and OTF (or other event unification at the end of
 * execution) we need to finalize APEX before MPI_Finalize() is called, so
 * that we can use MPI for the wrap-up.  We can override the weak MPI
 * implementation of Finalize, and do what we need to. */
#if defined(APEX_WITH_MPI) && !defined(HPX_HAVE_NETWORKING)
    int MPI_Init(int *argc, char ***argv) {
        int retval = PMPI_Init(argc, argv);
        int rank{0};
        int size{0};
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);
        apex::init("APEX MPI", rank, size);
        return retval;
    }
    int MPI_Init_thread( int *argc, char ***argv, int required, int *provided ) {
        int retval = PMPI_Init_thread(argc, argv, required, provided);
        int rank{0};
        int size{0};
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);
        apex::init("APEX MPI", rank, size);
        return retval;
    }
    int MPI_Finalize(void) {
        apex::finalize();
        int retval = PMPI_Finalize();
        apex::cleanup();
        return retval;
    }
    /* Define Fortran versions */
#define APEX_MPI_FINALIZE_TEMPLATE(_symbol) \
void  _symbol( MPI_Fint *ierr ) { \
    *ierr = MPI_Finalize(  ); \
}
    APEX_MPI_FINALIZE_TEMPLATE(mpi_finalize)
    APEX_MPI_FINALIZE_TEMPLATE(mpi_finalize_)
    APEX_MPI_FINALIZE_TEMPLATE(mpi_finalize__)
    APEX_MPI_FINALIZE_TEMPLATE(MPI_FINALIZE)
    APEX_MPI_FINALIZE_TEMPLATE(MPI_FINALIZE_)
    APEX_MPI_FINALIZE_TEMPLATE(MPI_FINALIZE__)
#endif

#if defined(APEX_WITH_MPI) || \
    (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI))
    /* Get the total bytes transferred, record it, and return it
       to be used for bandwidth calculation */
    inline double getBytesTransferred(int count, MPI_Datatype datatype, const char * function) {
        apex::in_apex prevent_memory_tracking;
        int typesize = 0;
        PMPI_Type_size( datatype, &typesize );
        double bytes = (double)(typesize) * (double)(count);
        std::string name("Bytes : ");
        name.append(function);
        apex::sample_value(name, bytes);
        return bytes;
    }
    inline double getBytesTransferred2(const int count, MPI_Datatype datatype, MPI_Comm comm, const char * function) {
        apex::in_apex prevent_memory_tracking;
        int typesize = 0;
        int commsize = 0;
        PMPI_Type_size( datatype, &typesize );
        PMPI_Comm_size( comm, &commsize );
        double bytes = (double)(typesize) * (double)(count) * (double)commsize;
        std::string name("Bytes : ");
        name.append(function);
        apex::sample_value(name, bytes);
        return bytes;
    }
    inline double getBytesTransferred3(const int * count, MPI_Datatype datatype, MPI_Comm comm, const char * function) {
        apex::in_apex prevent_memory_tracking;
        int typesize = 0;
        int commsize = 0;
        PMPI_Type_size( datatype, &typesize );
        PMPI_Comm_size( comm, &commsize );
        double bytes = 0;
        for(int i = 0 ; i < commsize ; i++) {
            bytes += ((double)(typesize) * (double)(count[i]));
        }
        std::string name("Bytes : ");
        name.append(function);
        apex::sample_value(name, bytes);
        return bytes;
    }
     inline void getBandwidth(double bytes, std::shared_ptr<apex::task_wrapper> task, const char * function) {
        apex::in_apex prevent_memory_tracking;
        if ((task != nullptr) && (task->prof != nullptr)) {
            std::string name("BW (Bytes/second) : ");
            name.append(function);
            apex::sample_value(name, bytes/task->prof->elapsed_seconds());
        }
    }
    /* There are also a handful of interesting function calls that HPX uses
       that we should measure when requested */
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
        int tag, MPI_Comm comm, MPI_Request *request) {
        /* Get the byte count */
        double bytes = getBytesTransferred(count, datatype, "MPI_Isend");
        /* start the timer */
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", bytes);
        /* sample the bytes */
        int retval = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
        MPI_STOP_TIMER
        /* record the bandwidth */
        //getBandwidth(bytes, p, "MPI_Isend");
        return retval;
    }
#define APEX_MPI_ISEND_TEMPLATE(_symbol) \
void  _symbol( void * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, \
    MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr ) { \
    MPI_Request local_request; \
    *ierr = MPI_Isend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request ); \
    *request = MPI_Request_c2f(local_request); \
}
    APEX_MPI_ISEND_TEMPLATE(mpi_isend)
    APEX_MPI_ISEND_TEMPLATE(mpi_isend_)
    APEX_MPI_ISEND_TEMPLATE(mpi_isend__)
    APEX_MPI_ISEND_TEMPLATE(MPI_ISEND)
    APEX_MPI_ISEND_TEMPLATE(MPI_ISEND_)
    APEX_MPI_ISEND_TEMPLATE(MPI_ISEND__)

    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request *request) {
        /* Get the byte count */
        double bytes = getBytesTransferred(count, datatype, "MPI_Irecv");
        MPI_START_TIMER
        apex::recordMetric("Recv Bytes", bytes);
        int retval = PMPI_Irecv(buf, count, datatype, source, tag, comm,
            request);
        MPI_STOP_TIMER
        /* record the bandwidth */
        //getBandwidth(bytes, p, "MPI_Irecv");
        return retval;
    }
#define APEX_MPI_IRECV_TEMPLATE(_symbol) \
void  _symbol( void * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * source, \
    MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr ) { \
    MPI_Request local_request; \
    *ierr = MPI_Irecv( buf, *count, MPI_Type_f2c(*datatype), *source, *tag, MPI_Comm_f2c(*comm), &local_request ); \
    *request = MPI_Request_c2f(local_request); \
}
    APEX_MPI_IRECV_TEMPLATE(mpi_irecv)
    APEX_MPI_IRECV_TEMPLATE(mpi_irecv_)
    APEX_MPI_IRECV_TEMPLATE(mpi_irecv__)
    APEX_MPI_IRECV_TEMPLATE(MPI_IRECV)
    APEX_MPI_IRECV_TEMPLATE(MPI_IRECV_)
    APEX_MPI_IRECV_TEMPLATE(MPI_IRECV__)

    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
        int tag, MPI_Comm comm){
        /* Get the byte count */
        double bytes = getBytesTransferred(count, datatype, "MPI_Send");
        /* start the timer */
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", bytes);
        /* sample the bytes */
        int retval = PMPI_Send(buf, count, datatype, dest, tag, comm);
        MPI_STOP_TIMER
        /* record the bandwidth */
        //getBandwidth(bytes, p, "MPI_Send");
        return retval;
    }
#define APEX_MPI_SEND_TEMPLATE(_symbol) \
void  _symbol( void * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, \
    MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * ierr ) { \
    *ierr = MPI_Send( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_SEND_TEMPLATE(mpi_send)
    APEX_MPI_SEND_TEMPLATE(mpi_send_)
    APEX_MPI_SEND_TEMPLATE(mpi_send__)
    APEX_MPI_SEND_TEMPLATE(MPI_SEND)
    APEX_MPI_SEND_TEMPLATE(MPI_SEND_)
    APEX_MPI_SEND_TEMPLATE(MPI_SEND__)

    int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Status *status){
        /* Get the byte count */
        double bytes = getBytesTransferred(count, datatype, "MPI_Recv");
        MPI_START_TIMER
        apex::recordMetric("Recv Bytes", bytes);
        int retval = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
        MPI_STOP_TIMER
        /* record the bandwidth */
        //getBandwidth(bytes, p, "MPI_Recv");
        return retval;
    }
#define APEX_MPI_RECV_TEMPLATE(_symbol) \
void  _symbol( void * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * source, \
    MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * status, MPI_Fint * ierr ) { \
    MPI_Status s; \
    *ierr = MPI_Recv( buf, *count, MPI_Type_f2c(*datatype), *source, *tag, MPI_Comm_f2c(*comm), &s ); \
    MPI_Status_c2f(&s, status); \
}
    APEX_MPI_RECV_TEMPLATE(mpi_recv)
    APEX_MPI_RECV_TEMPLATE(mpi_recv_)
    APEX_MPI_RECV_TEMPLATE(mpi_recv__)
    APEX_MPI_RECV_TEMPLATE(MPI_RECV)
    APEX_MPI_RECV_TEMPLATE(MPI_RECV_)
    APEX_MPI_RECV_TEMPLATE(MPI_RECV__)

    /* There are a handful of interesting Collectives! */
    inline int apex_measure_mpi_sync(MPI_Comm comm, const char * name, std::shared_ptr<apex::task_wrapper> parent) {
        APEX_UNUSED(name);
        //auto _p = start(std::string(name)+" (sync)");
        auto _p = new_task("MPI Collective Sync", UINTMAX_MAX, parent);
	    start(_p);
        int _retval = PMPI_Barrier(comm);
        stop(_p);
        return _retval;
    }
    int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(sendcount, sendtype, "MPI_Gather sendbuf");
        double rbytes = amIroot(comm, root) ? getBytesTransferred2(recvcount, recvtype, comm, "MPI_Gather recvbuf") : 0.0;
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf,
            recvcount, recvtype, root, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_GATHER_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcnt, MPI_Fint *sendtype, void * recvbuf, \
    MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Gather( sendbuf, *sendcnt, MPI_Type_f2c(*sendtype), recvbuf, *recvcount, \
        MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_GATHER_TEMPLATE(mpi_gather)
    APEX_MPI_GATHER_TEMPLATE(mpi_gather_)
    APEX_MPI_GATHER_TEMPLATE(mpi_gather__)
    APEX_MPI_GATHER_TEMPLATE(MPI_GATHER)
    APEX_MPI_GATHER_TEMPLATE(MPI_GATHER_)
    APEX_MPI_GATHER_TEMPLATE(MPI_GATHER__)

    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(count, datatype, "MPI_Allreduce sendbuf");
        double rbytes = getBytesTransferred2(count, datatype, comm, "MPI_Allreduce recvbuf");
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_ALLREDUCE_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, void * recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, \
    MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Allreduce( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_ALLREDUCE_TEMPLATE(mpi_allreduce)
    APEX_MPI_ALLREDUCE_TEMPLATE(mpi_allreduce_)
    APEX_MPI_ALLREDUCE_TEMPLATE(mpi_allreduce__)
    APEX_MPI_ALLREDUCE_TEMPLATE(MPI_ALLREDUCE)
    APEX_MPI_ALLREDUCE_TEMPLATE(MPI_ALLREDUCE_)
    APEX_MPI_ALLREDUCE_TEMPLATE(MPI_ALLREDUCE__)

    int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, int root, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(count, datatype, "MPI_Reduce sendbuf");
        double rbytes = amIroot(comm, root) ? getBytesTransferred2(count, datatype, comm, "MPI_Reduce recvbuf") : 0.0;
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_REDUCE_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, void * recvbuf, MPI_Fint *count, MPI_Fint *datatype, \
    MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Reduce( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), \
        MPI_Op_f2c(*op), *root, MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_REDUCE_TEMPLATE(mpi_reduce)
    APEX_MPI_REDUCE_TEMPLATE(mpi_reduce_)
    APEX_MPI_REDUCE_TEMPLATE(mpi_reduce__)
    APEX_MPI_REDUCE_TEMPLATE(MPI_REDUCE)
    APEX_MPI_REDUCE_TEMPLATE(MPI_REDUCE_)
    APEX_MPI_REDUCE_TEMPLATE(MPI_REDUCE__)

    int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
        MPI_Comm comm ) {
        //int commrank;
        //PMPI_Comm_rank(comm, &commrank);
        /* Get the byte count */
        double sbytes = getBytesTransferred(count, datatype, "MPI_Bcast");
        MPI_START_TIMER
        //if (root == commrank) {
            apex::recordMetric("Send Bytes", sbytes);
        //} else {
            //apex::recordMetric("Recv Bytes", sbytes);
        //}
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Bcast(buffer, count, datatype, root, comm );
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_BCAST_TEMPLATE(_symbol) \
void _symbol(void * buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, \
    MPI_Fint *comm, MPI_Fint *ierr) { \
    *ierr = MPI_Bcast( buffer, *count, MPI_Type_f2c(*datatype), *root, \
        MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_BCAST_TEMPLATE(mpi_bcast)
    APEX_MPI_BCAST_TEMPLATE(mpi_bcast_)
    APEX_MPI_BCAST_TEMPLATE(mpi_bcast__)
    APEX_MPI_BCAST_TEMPLATE(MPI_BCAST)
    APEX_MPI_BCAST_TEMPLATE(MPI_BCAST_)
    APEX_MPI_BCAST_TEMPLATE(MPI_BCAST__)

    int MPI_Startall(int count, MPI_Request array_of_requests[]) {
        MPI_START_TIMER
        int retval = PMPI_Startall(count, array_of_requests);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_DECL_LOCAL(mtype, _local) mtype *_local = nullptr
#define APEX_MPI_ALLOC_LOCAL(mtype, _local, size) _local = (mtype *) malloc(sizeof(mtype) * size)
#define APEX_MPI_DECL_ALLOC_LOCAL(mtype, _local, size) mtype * _local = (mtype *) malloc(sizeof(mtype) * size)
#define APEX_MPI_ASSIGN_VALUES(dest, src, size, func) { int i; for (i = 0; i < size; i++) dest[i] = func(src[i]); }
#define APEX_MPI_ASSIGN_STATUS_F2C(dest, src, size, func) { int i; for (i = 0; i < size; i++) func((MPI_Fint*)&((MPI_Status*)src)[i], &((MPI_Status*)dest)[i]); }
#define APEX_MPI_ASSIGN_STATUS_C2F(dest, src, size, func) { int i; for (i = 0; i < size; i++) func(&((MPI_Status*)src)[i], (MPI_Fint*)&((MPI_Status*)dest)[i]); }
#define APEX_MPI_FREE_LOCAL(_local) free(_local)
#define APEX_MPI_STARTALL_TEMPLATE(_symbol) \
void _symbol(MPI_Fint *count, MPI_Fint * array_of_requests, MPI_Fint *ierr) { \
    APEX_MPI_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count); \
    APEX_MPI_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c); \
    *ierr = MPI_Startall( *count, local_requests ); \
    APEX_MPI_ASSIGN_VALUES(array_of_requests, local_requests, *count, MPI_Request_c2f); \
    APEX_MPI_FREE_LOCAL(local_requests); \
}
    APEX_MPI_STARTALL_TEMPLATE(mpi_startall)
    APEX_MPI_STARTALL_TEMPLATE(mpi_startall_)
    APEX_MPI_STARTALL_TEMPLATE(mpi_startall__)
    APEX_MPI_STARTALL_TEMPLATE(MPI_STARTALL)
    APEX_MPI_STARTALL_TEMPLATE(MPI_STARTALL_)
    APEX_MPI_STARTALL_TEMPLATE(MPI_STARTALL__)

    int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(sendcount, sendtype, "MPI_Alltoall sendbuf");
        double rbytes = getBytesTransferred2(recvcount, recvtype, comm, "MPI_Alltoall recvbuf");
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_ALLTOALL_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void * recvbuf, \
MPI_Fint *recvcnt, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Alltoall(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcnt, \
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_ALLTOALL_TEMPLATE(mpi_alltoall)
    APEX_MPI_ALLTOALL_TEMPLATE(mpi_alltoall_)
    APEX_MPI_ALLTOALL_TEMPLATE(mpi_alltoall__)
    APEX_MPI_ALLTOALL_TEMPLATE(MPI_ALLTOALL)
    APEX_MPI_ALLTOALL_TEMPLATE(MPI_ALLTOALL_)
    APEX_MPI_ALLTOALL_TEMPLATE(MPI_ALLTOALL__)

    int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(sendcount, sendtype, "MPI_Allgather sendbuf");
        double rbytes = getBytesTransferred2(recvcount, recvtype, comm, "MPI_Allgather recvbuf");
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_ALLGATHER_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void * recvbuf, \
    MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Allgather( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, \
        *recvcount, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) ); \
}
/*
    APEX_MPI_ALLGATHER_TEMPLATE(mpi_allgather)
    APEX_MPI_ALLGATHER_TEMPLATE(mpi_allgather_)
    APEX_MPI_ALLGATHER_TEMPLATE(mpi_allgather__)
    APEX_MPI_ALLGATHER_TEMPLATE(MPI_ALLGATHER)
    APEX_MPI_ALLGATHER_TEMPLATE(MPI_ALLGATHER_)
    APEX_MPI_ALLGATHER_TEMPLATE(MPI_ALLGATHER__)
*/
    int MPI_Allgatherv(const void* buffer_send, int count_send, MPI_Datatype datatype_send,
        void* buffer_recv, const int* counts_recv, const int* displacements,
        MPI_Datatype datatype_recv, MPI_Comm communicator) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(count_send, datatype_send, "MPI_Allgatherv sendbuf");
        double rbytes = 0; //getBytesTransferred3(counts_recv, datatype_recv, communicator, "MPI_Allgatherv recvbuf");
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(communicator, __APEX_FUNCTION__, p);
        int retval = PMPI_Allgatherv(buffer_send, count_send, datatype_send,
            buffer_recv, counts_recv, displacements, datatype_recv, communicator);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_ALLGATHERV_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void * recvbuf, \
    MPI_Fint * recvcounts, MPI_Fint * displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Allgatherv( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, recvcounts, \
        displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_ALLGATHERV_TEMPLATE(mpi_allgatherv)
    APEX_MPI_ALLGATHERV_TEMPLATE(mpi_allgatherv_)
    APEX_MPI_ALLGATHERV_TEMPLATE(mpi_allgatherv__)
    APEX_MPI_ALLGATHERV_TEMPLATE(MPI_ALLGATHERV)
    APEX_MPI_ALLGATHERV_TEMPLATE(MPI_ALLGATHERV_)
    APEX_MPI_ALLGATHERV_TEMPLATE(MPI_ALLGATHERV__)

    int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs,
        MPI_Datatype recvtype, int root, MPI_Comm comm) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(sendcount, sendtype, "MPI_Gatherv sendbuf");
        double rbytes = 0; //amIroot(comm, root) ? getBytesTransferred3(recvcounts, recvtype, comm, "MPI_Gatherv recvbuf") : 0;
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_GATHERV_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcnt, MPI_Fint *sendtype, void * recvbuf, MPI_Fint * recvcnts, \
    MPI_Fint * displs, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr) { \
    APEX_CREATE_CONVERTERRS \
    if (sendbuf == *(mpi_predef_in_place())) { sendbuf = MPI_IN_PLACE; } \
    if (sendbuf == *(mpi_predef_bottom())) { sendbuf = MPI_BOTTOM; } \
    if (recvbuf == *(mpi_predef_bottom())) { recvbuf = MPI_BOTTOM; } \
    *ierr = MPI_Gatherv( sendbuf, *sendcnt, MPI_Type_f2c(*sendtype), recvbuf, recvcnts, displs, \
    MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) ); \
}
    APEX_MPI_GATHERV_TEMPLATE(mpi_gatherv)
    APEX_MPI_GATHERV_TEMPLATE(mpi_gatherv_)
    APEX_MPI_GATHERV_TEMPLATE(mpi_gatherv__)
    APEX_MPI_GATHERV_TEMPLATE(MPI_GATHERV)
    APEX_MPI_GATHERV_TEMPLATE(MPI_GATHERV_)
    APEX_MPI_GATHERV_TEMPLATE(MPI_GATHERV__)

  int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int source, int recvtag, MPI_Comm comm, MPI_Status * status) {
        /* Get the byte count */
        double sbytes = getBytesTransferred(sendcount, sendtype, "MPI_Sendrecv sendbuf");
        double rbytes = getBytesTransferred(recvcount, recvtype, "MPI_Sendrecv recvbuf");
        MPI_START_TIMER
        apex::recordMetric("Send Bytes", sbytes);
        apex::recordMetric("Recv Bytes", rbytes);
        apex_measure_mpi_sync(comm, __APEX_FUNCTION__, p);
        int retval = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
                 source, recvtag, comm, status);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_SENDRECV_TEMPLATE(_symbol) \
void _symbol(void * sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, MPI_Fint *dest, \
    MPI_Fint *sendtag, void * recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, \
    MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint * status, MPI_Fint *ierr) { \
    MPI_Status local_status; \
    *ierr = MPI_Sendrecv( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), *dest, *sendtag, \
        recvbuf, *recvcount, MPI_Type_f2c(*recvtype), *source, *recvtag, MPI_Comm_f2c(*comm), &local_status ); \
    MPI_Status_c2f(&local_status, status); \
}

    /* There are a handful of interesting Synchronization functions */
    int MPI_Waitall(int count, MPI_Request array_of_requests[],
        MPI_Status array_of_statuses[]) {
        MPI_START_TIMER
        int retval = PMPI_Waitall(count, array_of_requests, array_of_statuses);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_WAITALL_TEMPLATE(_symbol) \
void _symbol(MPI_Fint *count, MPI_Fint * array_of_requests, MPI_Fint * array_of_statuses, \
    MPI_Fint *ierr) {\
    APEX_MPI_DECL_LOCAL(MPI_Status, local_statuses); \
    APEX_MPI_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count); \
    /* *CWL* - keep an eye on this. Make sure MPI_F_STATUSES_IGNORE is portable. */ \
    APEX_CREATE_CONVERTERRS \
    if (((MPI_Status*)(array_of_statuses)) != *(mpi_predef_statuses_ignore())) { \
        APEX_MPI_ALLOC_LOCAL(MPI_Status, local_statuses, *count); \
    } \
    APEX_MPI_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c); \
    if (((MPI_Status*)(array_of_statuses)) != *(mpi_predef_statuses_ignore())) { \
        APEX_MPI_ASSIGN_STATUS_F2C(local_statuses, array_of_statuses, *count, MPI_Status_f2c); \
    } \
     \
    if (((MPI_Status*)(array_of_statuses)) != *(mpi_predef_statuses_ignore())) { \
        *ierr = MPI_Waitall( *count, local_requests, local_statuses ); \
    } else { \
        /* Remember, we're invoking the C interface now. */ \
        *ierr = MPI_Waitall( *count, local_requests, MPI_STATUSES_IGNORE ); \
    } \
    APEX_MPI_ASSIGN_VALUES(array_of_requests, local_requests, *count, MPI_Request_c2f); \
     \
    if (((MPI_Status*)(array_of_statuses)) != *(mpi_predef_statuses_ignore())) { \
        APEX_MPI_ASSIGN_STATUS_C2F(array_of_statuses, local_statuses, *count, MPI_Status_c2f); \
    } \
    APEX_MPI_FREE_LOCAL(local_requests); \
    if (((MPI_Status*)(array_of_statuses)) != *(mpi_predef_statuses_ignore())) { \
        APEX_MPI_FREE_LOCAL(local_statuses); \
    } \
}
    APEX_MPI_WAITALL_TEMPLATE(mpi_waitall)
    APEX_MPI_WAITALL_TEMPLATE(mpi_waitall_)
    APEX_MPI_WAITALL_TEMPLATE(mpi_waitall__)
    APEX_MPI_WAITALL_TEMPLATE(MPI_WAITALL)
    APEX_MPI_WAITALL_TEMPLATE(MPI_WAITALL_)
    APEX_MPI_WAITALL_TEMPLATE(MPI_WAITALL__)

    int MPI_Wait(MPI_Request *request, MPI_Status *status) {
        MPI_START_TIMER
        int retval = PMPI_Wait(request, status);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_WAIT_TEMPLATE(_symbol) \
void _symbol(MPI_Fint * request, MPI_Fint * status, MPI_Fint * ierr) { \
    MPI_Request local_request = MPI_Request_f2c(*request); \
    MPI_Status local_status; \
    *ierr = MPI_Wait( &local_request, &local_status ); \
    *request = MPI_Request_c2f(local_request); \
    MPI_Status_c2f(&local_status, status); \
}
    APEX_MPI_WAIT_TEMPLATE(mpi_wait)
    APEX_MPI_WAIT_TEMPLATE(mpi_wait_)
    APEX_MPI_WAIT_TEMPLATE(mpi_wait__)
    APEX_MPI_WAIT_TEMPLATE(MPI_WAIT)
    APEX_MPI_WAIT_TEMPLATE(MPI_WAIT_)
    APEX_MPI_WAIT_TEMPLATE(MPI_WAIT__)

    int MPI_Barrier(MPI_Comm comm) {
        MPI_START_TIMER
        auto _p = apex::new_task("MPI Collective Sync");
	    apex::start(_p);
        int retval = PMPI_Barrier(comm);
	    apex::stop(_p);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_BARRIER_TEMPLATE(_symbol) \
void _symbol(MPI_Fint *comm, MPI_Fint * ierr) { \
    *ierr = MPI_Barrier( MPI_Comm_f2c(*comm) ); \
}
#if 0
    APEX_MPI_BARRIER_TEMPLATE(mpi_barrier)
    APEX_MPI_BARRIER_TEMPLATE(mpi_barrier_)
    APEX_MPI_BARRIER_TEMPLATE(mpi_barrier__)
    APEX_MPI_BARRIER_TEMPLATE(MPI_BARRIER)
    APEX_MPI_BARRIER_TEMPLATE(MPI_BARRIER_)
    APEX_MPI_BARRIER_TEMPLATE(MPI_BARRIER__)

    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
        MPI_START_TIMER
        int retval = PMPI_Test(request, flag, status);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_TEST_TEMPLATE(_symbol) \
void _symbol(MPI_Fint * request, MPI_Fint * flag, MPI_Fint * status, MPI_Fint *ierr) { \
    MPI_Status local_status; \
    MPI_Request local_request = MPI_Request_f2c(*request); \
    *ierr = MPI_Test( &local_request, flag, &local_status ); \
    *request = MPI_Request_c2f(local_request); \
    MPI_Status_c2f(&local_status, status); \
}
    APEX_MPI_TEST_TEMPLATE(mpi_test)
    APEX_MPI_TEST_TEMPLATE(mpi_test_)
    APEX_MPI_TEST_TEMPLATE(mpi_test__)
    APEX_MPI_TEST_TEMPLATE(MPI_TEST)
    APEX_MPI_TEST_TEMPLATE(MPI_TEST_)
    APEX_MPI_TEST_TEMPLATE(MPI_TEST__)

    int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx,
        int *flag, MPI_Status *status) {
        MPI_START_TIMER
        int retval = PMPI_Testany(count, array_of_requests, indx, flag, status);
        MPI_STOP_TIMER
        return retval;
    }
#define APEX_MPI_TESTANY_TEMPLATE(_symbol) \
void _symbol(MPI_Fint *count, MPI_Fint * array_of_requests, MPI_Fint * index, \
    MPI_Fint * flag, MPI_Fint * status, MPI_Fint *ierr) { \
    MPI_Status local_status; \
    APEX_MPI_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count); \
    APEX_MPI_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c); \
    *ierr  = MPI_Testany( *count, local_requests, index, flag, &local_status ); \
    APEX_MPI_ASSIGN_VALUES(array_of_requests, local_requests, *count, MPI_Request_c2f); \
    MPI_Status_c2f(&local_status, status); \
    APEX_MPI_FREE_LOCAL(local_requests); \
    /* Increment the C index before returning it as a Fortran index as \
        [0..N-1] => [1..N] array indexing differs in C and Fortran */ \
    if ((*index != MPI_UNDEFINED) && (*index >= 0)) \
        (*index)++; \
}
    APEX_MPI_TESTANY_TEMPLATE(mpi_testany)
    APEX_MPI_TESTANY_TEMPLATE(mpi_testany_)
    APEX_MPI_TESTANY_TEMPLATE(mpi_testany__)
    APEX_MPI_TESTANY_TEMPLATE(MPI_TESTANY)
    APEX_MPI_TESTANY_TEMPLATE(MPI_TESTANY_)
    APEX_MPI_TESTANY_TEMPLATE(MPI_TESTANY__)
#endif
#endif

} // extern "C"


