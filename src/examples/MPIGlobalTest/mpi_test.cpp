#include <mpi.h>
#include <limits.h>
#include "apex.hpp"
#include "apex_global.h"

#define WORKTAG 1
#define DIETAG 2

#define unit_of_work_t int*
#define unit_result_t int*

/* Local functions */

static void master(void);
static void worker(void);
static unit_of_work_t get_next_work_item(void);
static void process_results(unit_result_t result);
static unit_result_t do_work(unit_of_work_t work);

static int dummy = 0;

int
main(int argc, char **argv)
{
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
  if (myrank == 0) {
    master();
  } else {
    worker();
  }

  /* Shut down MPI */

  MPI_Finalize();
  apex::finalize();
  return 0;
}


static void
master(void)
{
  int ntasks, rank;
  unit_of_work_t work = NULL;
  unit_result_t result = NULL;
  MPI_Status status;
  void * profiler = apex::start((void*)(master));

  /* Find out how many processes there are in the default
     communicator */

  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  /* Seed the worker; send one unit of work to each worker. */

  for (rank = 1; rank < ntasks; ++rank) {

    /* Find the next item of work to do */

    work = get_next_work_item();

    /* Send it to each rank */

    MPI_Send(work,             /* message buffer */
             1,                 /* one data item */
             MPI_INT,           /* data item is an integer */
             rank,              /* destination process rank */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
  }

  // do some work myself
  work = get_next_work_item();
  result = do_work(work);

  /* Loop over getting new work requests until there is no more work
     to be done */

  work = get_next_work_item();
  while (work != NULL) {

    /* Receive results from a worker */

    MPI_Recv(result,           /* message buffer */
             1,                 /* one data item */
             MPI_INT,        /* of type double real */
             MPI_ANY_SOURCE,    /* receive from any sender */
             MPI_ANY_TAG,       /* any type of message */
             MPI_COMM_WORLD,    /* default communicator */
             &status);          /* info about the received message */

    /* Send the worker a new work unit */

    MPI_Send(work,             /* message buffer */
             1,                 /* one data item */
             MPI_INT,           /* data item is an integer */
             status.MPI_SOURCE, /* to who we just received from */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */

    // do some work myself
    work = get_next_work_item();
	if (work != NULL) {
      result = do_work(work);

    /* Get the next unit of work to be done */

      work = get_next_work_item();
	}
  }

  /* There's no more work to be done, so receive all the outstanding
     results from the worker. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Recv(result, 1, MPI_INT, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }

  /* Tell all the worker to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
  apex::stop(profiler);
}


static void 
worker(void)
{
  unit_of_work_t work = NULL;
  unit_result_t results = NULL;
  MPI_Status status;
  void * profiler = apex::start((void*)(worker));

  while (1) {

    /* Receive a message from the master */

    MPI_Recv(work, 1, MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    /* Check the tag of the received message. */

    if (status.MPI_TAG == DIETAG) {
      return;
    }

    /* Do the work */

    unit_result_t result = do_work(work);

    /* Send the result back */

    MPI_Send(result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  apex::stop(profiler);
}


static unit_of_work_t 
get_next_work_item(void)
{
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a worker. */
	static int data[] = {1,2,3,4,5,6,7,8,9,10};
	static int index = -1;
	if (++index < 10) return &(data[index]);
	return NULL;
}

#define UNUSED(x) (void)(x)

static void 
process_results(unit_result_t result)
{
  /* Fill in with whatever is relevant to process the results returned
     by the worker */
	 UNUSED(result);
	return ;
}


static unit_result_t
do_work(unit_of_work_t work)
{
  void * profiler = apex::start((void*)(do_work));
  int * mywork = (int*)(work);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int i;
  for (i = 0 ; i < 500000000 ; i++) {
    dummy = dummy * (dummy + 1);
	if (dummy > (INT_MAX >> 1)) {
	  dummy = 1;
	}
  }
  /* Fill in with whatever is necessary to process the work and
     generate a result */
  apex::stop(profiler);
  return &dummy;
}

