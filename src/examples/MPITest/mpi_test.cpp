#include <mpi.h>
#include <unistd.h>
#include "apex_api.hpp"

#define WORKTAG 1
#define DIETAG 2

#define DEBUG_MSG printf("%d:%d - %s\n", myrank, commsize, __func__)
#define DEBUG_STATUS(MSG) printf("%d:%d - %s\n", myrank, commsize, (MSG))

/* Local functions */

static void root_node(void);
static void worker_node(void);
static int get_next_work_item(void);
static int do_work(int work);

static int dummy = 0;
static int myrank = -1;
static int commsize = -1;
static int nworkitems = -1;
constexpr int items_per_rank = 1;

int main(int argc, char **argv) {
  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  /* Find out my identity in the default communicator */
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  nworkitems = commsize * items_per_rank;
  DEBUG_MSG;
  apex::init("MPI TEST", myrank, commsize);
  apex::profiler* p = apex::start(__func__);
  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == 0) {
    root_node();
  } else {
    worker_node();
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


static void root_node(void) {
  DEBUG_MSG;
  int work = 0;
  int result = 0;
  int rank = 0;
  int outstanding = 0;
  MPI_Status status;
  APEX_SCOPED_TIMER;

  /* Seed the worker_nodes; send one unit of work to each worker_node. */

  for (rank = 1; rank < commsize; ++rank) {

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

    /* Receive results from a worker_node */

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

        /* Send the worker_node a new work unit */
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
     results from the worker_node. */

  while (outstanding > 0) {
    DEBUG_STATUS("receiving result");
    MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    apex::recv(status.MPI_TAG, sizeof(int), status.MPI_SOURCE, 0);
    outstanding = outstanding - 1;
  }

  /* Tell all the worker_node to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < commsize; ++rank) {
    DEBUG_STATUS("sending exit");
    apex::send(DIETAG, sizeof(int), rank);
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
  DEBUG_STATUS("exiting");
}

static void worker_node(void) {
  DEBUG_MSG;
  int work = 0;
  MPI_Status status;
  APEX_SCOPED_TIMER;

  while (1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    /* Receive a message from the root_node */

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
  return;
}

int * get_work_items() {
  APEX_SCOPED_TIMER;
  int * tmp = (int*)(malloc(sizeof(int)*nworkitems));
  for (int i = 0 ; i < nworkitems ; i++) {
    tmp[i] = i+1;
  }
  return tmp;
}

static int get_next_work_item(void)
{
  APEX_SCOPED_TIMER;
  DEBUG_MSG;
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a worker_node. */
    static int * data = get_work_items();
    static int index = -1;
    int value = 0;
    if (++index < nworkitems) {
        value = data[index];
    }
    return value;
}

static int do_work(int work)
{
  APEX_SCOPED_TIMER;
  DEBUG_MSG;
  int jiffies = work * 10;
  usleep(jiffies);
  dummy = dummy + work;
  /* Fill in with whatever is necessary to process the work and
     generate a result */
    return dummy;
}

