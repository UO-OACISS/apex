#include <mpi.h>
#include "apex_api.hpp"

#define WORKTAG 1
#define DIETAG 2

#define DEBUG_MSG printf("%d:%d - %s\n", myrank, commsize, __func__)
#define DEBUG_STATUS(MSG) printf("%d:%d - %s\n", myrank, commsize, (MSG))

/* Local functions */

static void master(void);
static void worker(void);
static int get_next_work_item(void);
static int do_work(int work);

static int dummy = 0;
static int myrank = -1;
static int commsize = -1;

int main(int argc, char **argv) {
  DEBUG_MSG;

  /* Initialize MPI */

  /*
  int required, provided;
  required = MPI_THREAD_FUNNELED;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    printf ("Your MPI installation doesn't allow multiple threads. Exiting.\n");
        exit(0);
  }
  */
  MPI_Init(&argc, &argv);

  /* Find out my identity in the default communicator */

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  apex::init("MPI TEST", myrank, commsize);
  apex::profiler* p = apex::start((apex_function_address)(main));
  MPI_Barrier(MPI_COMM_WORLD);
  if (commsize < 2) {
    printf("Please run with 2 or more MPI ranks.\n");
    apex::finalize();
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  if (myrank == 0) {
    master();
  } else {
    worker();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  apex::stop(p);

  /* Shut down MPI */

  DEBUG_STATUS("finalizing APEX");
  apex::finalize();
  DEBUG_STATUS("finalizing MPI");
  MPI_Finalize();
  return 0;
}


static void master(void) {
  DEBUG_MSG;
  int ntasks, rank;
  int work = 0;
  int result = 0;
  int outstanding = 0;
  MPI_Status status;
  apex::profiler* p = apex::start((apex_function_address)(master));

  /* Find out how many processes there are in the default
     communicator */

  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  /* Seed the worker; send one unit of work to each worker. */

  for (rank = 1; rank < ntasks; ++rank) {

    /* Find the next item of work to do */
    work = get_next_work_item();

    if (work != 0) {
        DEBUG_STATUS("sending work");
        /* Send it to each rank */
        apex::send(WORKTAG, sizeof(int), rank);
        MPI_Send(&work,              /* message buffer */
                1,                 /* one data item */
                MPI_INT,           /* data item is an integer */
                rank,              /* destination process rank */
                WORKTAG,           /* user chosen message tag */
                MPI_COMM_WORLD);   /* default communicator */
        outstanding = outstanding + 1;
    }
  }

  // do some work ourselves.
  work = get_next_work_item();
  if (work != 0) {
    do_work(work);
  }

  /* Loop over getting new work requests until there is no more work
     to be done */

  while (work != 0) {

    /* Receive results from a worker */

    while (outstanding > 0) {
        DEBUG_STATUS("receiving result");
        MPI_Recv(&result,            /* message buffer */
                1,                 /* one data item */
                MPI_INT,        /* of type double real */
                MPI_ANY_SOURCE,    /* receive from any sender */
                MPI_ANY_TAG,       /* any type of message */
                MPI_COMM_WORLD,    /* default communicator */
                &status);          /* info about the received message */
        apex::recv(status.MPI_TAG, sizeof(int), status.MPI_SOURCE, 0);
        outstanding = outstanding - 1;

        /* Send the worker a new work unit */
        work = get_next_work_item();

        if (work != 0) {
            DEBUG_STATUS("sending work");
            apex::send(WORKTAG, sizeof(int), status.MPI_SOURCE);
            MPI_Send(&work,              /* message buffer */
                    1,                 /* one data item */
                    MPI_INT,           /* data item is an integer */
                    status.MPI_SOURCE, /* to who we just received from */
                    WORKTAG,           /* user chosen message tag */
                    MPI_COMM_WORLD);   /* default communicator */
            outstanding = outstanding + 1;
        }
    }

    /* Get the next unit of work to be done */
    work = get_next_work_item();
    if (work != 0) {
      do_work(work);
    }
  }

  /* There's no more work to be done, so receive all the outstanding
     results from the worker. */

  while (outstanding > 0) {
    DEBUG_STATUS("receiving result");
    MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    apex::recv(status.MPI_TAG, sizeof(int), status.MPI_SOURCE, 0);
    outstanding = outstanding - 1;
  }

  /* Tell all the worker to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < ntasks; ++rank) {
    DEBUG_STATUS("sending exit");
    apex::send(DIETAG, sizeof(int), rank);
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
  apex::stop(p);
  DEBUG_STATUS("exiting");
}

static void worker(void) {
  DEBUG_MSG;
  int work = 0;
  MPI_Status status;
  apex::profiler* p = apex::start((apex_function_address)(worker));

  while (1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    /* Receive a message from the master */

    DEBUG_STATUS("receiving work");
    MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);
    apex::recv(status.MPI_TAG, sizeof(int), status.MPI_SOURCE, 0);

    /* Check the tag of the received message. */

    if (status.MPI_TAG == DIETAG || work == 0) {
        DEBUG_STATUS("exiting");
        break;
    }

    /* Do the work */

    int result = do_work(work);

    /* Send the result back */

    DEBUG_STATUS("sending result");
    apex::send(0, sizeof(int), 0);
    MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  apex::stop(p);
  return;
}


static int get_next_work_item(void)
{
  DEBUG_MSG;
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a worker. */
    static int data[] = {1,2,3,4,5,6,7,8,9,10};
    static int index = -1;
    if (++index < 10) return (data[index]);
    return 0;
}

static int do_work(int work)
{
  DEBUG_MSG;
  apex::profiler* p = apex::start((apex_function_address)(do_work));
  //sleep(*mywork);
  dummy = dummy + work;
  /* Fill in with whatever is necessary to process the work and
     generate a result */
  apex::stop(p);
    return dummy;
}

