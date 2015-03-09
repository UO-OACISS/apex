#include <mpi.h>
#include <limits.h>
#include "apex.hpp"
#include "apex_global.h"

#define WORKTAG 1
#define DIETAG 2
#define TOTAL_WORK 10

#define unit_of_work_t int
#define unit_result_t int

/* Local functions */

static void master(void);
static void worker(void);
static void get_next_work_item(int work[], int rank, int ntasks);
static unit_result_t do_work(unit_of_work_t work);

using namespace std;

static int dummy = 0;
int * data;

void initialize_work(int * data, int size) {
    int index = 0;
    for (index = 0 ; index < size ; index++) { data[index] = index; }
    return;
}

int main(int argc, char **argv) {
  int myrank;

  /* Initialize MPI */

  int required, provided;
  required = MPI_THREAD_SERIALIZED;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    printf ("Your MPI installation doesn't allow multiple threads to communicate. Exiting.\n");
    exit(0);
  }
  apex::init(argc, argv, "MPI TEST");
  apex_global_setup((apex_function_address)(do_work));

  /* Find out my identity in the default communicator */

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  apex::set_node_id(myrank);

  int size = TOTAL_WORK;
  data = (int*)(malloc(size * sizeof(int)));

  if (myrank == 0) {
    initialize_work(data, size);
    MPI_Bcast(data, size, MPI_INT, 0, MPI_COMM_WORLD);
    master();
  } else {
    MPI_Bcast(data, size, MPI_INT, 0, MPI_COMM_WORLD);
    worker();
  }

  /* Shut down MPI */

  apex_global_teardown(); // do this before MPI_Finalize
  apex::finalize();
  MPI_Finalize();
  return 0;
}

static void master(void) {
  int ntasks, rank;
  unit_of_work_t work[2] = {0,0};
  unit_result_t result = 0;
  MPI_Status status;
  apex::profiler * p = apex::start((apex_function_address)(master));

  /* Find out how many processes there are in the default
     communicator */

  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  /* Seed the worker; send one unit of work to each worker. */

  for (rank = 1; rank < ntasks; ++rank) {

    /* Find the next item of work to do */

    get_next_work_item(work, rank, ntasks);

    /* Send it to each rank */

    MPI_Send(work,             /* message buffer */
             2,                 /* one data item */
             MPI_INT,           /* data item is an integer */
             rank,              /* destination process rank */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
  }

  /* There's no more work to be done, so receive all the outstanding
     results from the worker. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }

  /* Tell all the worker to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
  apex::stop(p);
}

static void worker(void) {
  unit_of_work_t work[] = {0,0};
  MPI_Status status;
  apex::profiler * p = apex::start((apex_function_address)(worker));

  while (1) {

    /* Receive a message from the master */

    MPI_Recv(&work, 2, MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    /* Check the tag of the received message. */

    if (status.MPI_TAG == DIETAG) {
      return;
    }

    /* Do the work */

    unit_result_t result = 0;
    int workindex;
    for (workindex = work[0] ; workindex < work[1] ; workindex++) {
      result += do_work(workindex);
    }

    /* Send the result back */

    MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  apex::stop(p);
}


static void get_next_work_item(int work[], int rank, int ntasks) {
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a worker. */
  int range = TOTAL_WORK / ntasks;
  int lower = range * rank;
  int upper = min((lower+range),TOTAL_WORK);
  work[0] = lower;
  work[1] = upper;
  return;
}

static unit_result_t do_work(unit_of_work_t work) {
  apex::profiler * p = apex::start((apex_function_address)(do_work));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int i, multiplier;
  // introduce an imbalance
  //printf("%d working...\n\n", rank);
    for (multiplier = 0 ; multiplier < ((rank % 2) + 1) ; multiplier++) {
      for (i = 0 ; i < 500000000 ; i++) {
        dummy = dummy * (dummy + data[work]);
        if (dummy > (INT_MAX >> 1)) {
          dummy = 1;
        }
      }
    }
  /* Fill in with whatever is necessary to process the work and
     generate a result */
  apex::stop(p);
  return dummy;
}

