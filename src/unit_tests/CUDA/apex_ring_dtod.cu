#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include "util_cuda.hpp"
#include "util_cuda.cu"
#include "util_mpi.hpp"
#include "apex_api.hpp"

#define MOD(x,n) ((x) % (n))

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)


int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, mpi_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    apex::init("GPU DtoD test", rank, mpi_size);

    // sync all processors at the beginning
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_Request recv_request;
    MPI_Request send_request;
    MPI_Status status;

    int left_neighbor = MOD((rank-1 + mpi_size), mpi_size);
    int right_neighbor = MOD((rank+1 + mpi_size), mpi_size);

    // number of G2s
    int niter = 2;

    size_t n_elems = 8388608; // 2 ^ 23
    float* G2 = nullptr;
    float* G4 = nullptr;
    float* sendbuff_G2 = nullptr;
    float* recvbuff_G2 = nullptr;

    int count;
    RUNTIME_API_CALL(cudaGetDeviceCount(&count));
    printf("Rank %d:%d Found %d devices...\n", rank, mpi_size, count);
    printf("Rank %d:%d using devices %d\n", rank, mpi_size, MOD(rank,count));
    RUNTIME_API_CALL(cudaSetDevice(MOD(rank,count)));
    G2 = allocate_on_device<float>(n_elems);
    G4 = allocate_on_device<float>(n_elems);
    sendbuff_G2 = allocate_on_device<float>(n_elems);
    recvbuff_G2 = allocate_on_device<float>(n_elems);

    double start_time, end_time;
    // sync all processors at the end
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }
    for(int i = 0; i < niter; i++)
    {
        // generate G2 and fill some value in
        generateG2(G2, rank, n_elems);
        update_local_G4(G2, G4, rank, n_elems);

        // get ready for send
        CudaMemoryCopy(sendbuff_G2, G2, n_elems);
        int send_tag = 1 + rank;
        send_tag = 1 + MOD(send_tag-1, INT_MAX); // just to be safe, INT_MAX is largest tag value
        for(int icount=0; icount < (mpi_size-1); icount++)
        {
            // encode the originator rank in the message tag as tag = 1 + originator_irank
            int originator_irank = MOD(((rank-1)-icount + 2*mpi_size), mpi_size);
            int recv_tag = 1 + originator_irank;
            recv_tag = 1 + MOD(recv_tag-1, INT_MAX); // just to be safe, then 1 <= tag <= INT_MAX

            MPI_CHECK(MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, recv_tag, MPI_COMM_WORLD, &recv_request));
            MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, send_tag, MPI_COMM_WORLD, &send_request));

            MPI_CHECK(MPI_Wait(&recv_request, &status));
            CudaMemoryCopy(G2, recvbuff_G2, n_elems);
            update_local_G4(G2, G4, rank, n_elems);
            MPI_CHECK(MPI_Wait(&send_request, &status)); // wait for sendbuf_G2 to be available again

            // get ready for send
            CudaMemoryCopy(sendbuff_G2, G2, n_elems);
            send_tag = recv_tag;
        }
    }

    if (rank == 0)
    {
        end_time = MPI_Wtime();
        printf("Total time spent on %d iteration: %lf \n", niter, (end_time - start_time));
        printf("Number of ranks: %d, number of float elements %d \n", mpi_size, n_elems);
        printf("Average time spent on 1 iteration: %lf \n", (end_time - start_time) / niter);
    }

    // sync all processors at the end
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    apex::finalize();
    MPI_CHECK(MPI_Finalize());
}