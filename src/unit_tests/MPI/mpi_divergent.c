/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include <stdio.h>
#include "apex.h"

void foo0(int myid);
void foo1(int myid);
void foo2(int myid);
void foo3(int myid);

void foo0(int myid) {
  apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, (void*)__func__);
  usleep(100);
  apex_stop(profiler);
  return;
}

void foo1(int myid) {
  apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, (void*)__func__);
  usleep(100);
  foo0(myid);
  apex_stop(profiler);
  return;
}

void foo2(int myid) {
  apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, (void*)__func__);
  usleep(100);
  foo1(myid);
  apex_stop(profiler);
  return;
}

void foo3(int myid) {
  apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, (void*)__func__);
  usleep(100);
  foo2(myid);
  apex_stop(profiler);
  return;
}

int main(int argc,char *argv[])
{
    int    namelen;
    int    myid, numprocs;
    char   processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);
    apex_init("apex_start unit test", myid, numprocs);
    apex_set_use_screen_output(1);
    apex_profiler_handle profiler = apex_start(APEX_NAME_STRING, (void*)__func__);

    fprintf(stdout,"Process %d of %d is on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    switch(myid % 4) {
        case 3:
            foo3(myid);
        case 2:
            foo2(myid);
        case 1:
            foo1(myid);
        default:
            foo0(myid);
            break;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    apex_stop(profiler);
    apex_finalize();
    MPI_Finalize();
    apex_cleanup();
    return 0;
}
