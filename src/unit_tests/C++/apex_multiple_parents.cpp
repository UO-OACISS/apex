#include <thread>
#include <future>
#include <iostream>
#include <stdlib.h>
#include "apex_api.hpp"
#include <atomic>
#ifdef APEX_ENABLE_MPI
#include "mpi.h"
#endif

int parent (int in, std::shared_ptr< apex::task_wrapper > this_task) {
    apex::start(this_task);
    usleep(in*100);
    apex::stop(this_task);
    std::cout << "p1" << std::endl;
    return in;
}

int child (int in, std::shared_ptr< apex::task_wrapper > this_task) {
    apex::start(this_task);
    usleep(in*100);
    std::cout << "c" << std::endl;
    apex::stop(this_task);
    return in;
}

int main(int argc, char *argv[]) {
    int comm_rank = 0;
    int comm_size = 1;
#ifdef APEX_ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    std::cout << "APP: rank " << comm_rank << " of " << comm_size << std::endl;
#endif
    apex::init("apex_multiple_parents_cpp unit test", comm_rank, comm_size);
    apex::scoped_timer foo(__func__);

    int task_id{0};
	// called from the parent, not when the child is spawned!
    std::vector<std::shared_ptr<apex::task_wrapper>> parents;
    auto p1 = apex::new_task(std::string("parent1"), ++task_id);
    auto f1 = std::async(std::launch::async, parent, task_id, p1);
    parents.push_back(p1);

    auto p2 = apex::new_task(std::string("parent2"), ++task_id);
    auto f2 = std::async(std::launch::async, parent, task_id, p2);
    parents.push_back(p2);

    auto p3 = apex::new_task(std::string("parent3"), ++task_id);
    auto f3 = std::async(std::launch::async, parent, task_id, p3);
    parents.push_back(p3);

    auto p4 = apex::new_task(std::string("parent4"), ++task_id);
    auto f4 = std::async(std::launch::async, parent, task_id, p4);
    parents.push_back(p4);

    auto c = apex::new_task(std::string("child"), ++task_id, parents);
    auto f5 = std::async(std::launch::async, child, (int)task_id, c);

    int result = f1.get() + f2.get() + f5.get() + f3.get() + f4.get();
    std::cout << "sum is " << result << " (valid value: 6)" << std::endl;
	foo.stop();
    //apex::finalize();
#ifdef APEX_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return 0;
}

