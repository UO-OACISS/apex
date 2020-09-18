//
// Created by Wei, Weile on 10/31/19.
//

#ifndef MPI_CUDA_UTIL_MPI_HPP
#define MPI_CUDA_UTIL_MPI_HPP

#include <cassert>

#define MPI_CHECK(stmt)                                          \
do {                                                             \
   int mpi_errno = (stmt);                                       \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
} while (0)

#endif //MPI_CUDA_UTIL_MPI_HPP