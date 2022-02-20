#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <omp.h>
#include "apex_api.hpp"
#if defined(APEX_HAVE_MPI)
#include "mpi.h"
#endif

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s Line %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(x));}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

__global__ void montecarlo(float* pt1, float* pt2, int* result, int total_threads, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < n) {
    float sq = pt1[tid]*pt1[tid] + pt2[tid]*pt2[tid];
    if (sq < 1) {
      result[tid] = 1;
    }
    else {
      result[tid] = 0;
    }
    tid += total_threads;
  }
}

int main(int argc, char * argv[]) {
#if defined(APEX_HAVE_MPI)
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  apex::init("apex::cuda PI test", rank, size);
#else
  apex::init("apex cuda PI test", 0, 1);
#endif
  apex::apex_options::use_screen_output(true);
  //omp_set_num_threads(2);

  int num_darts = 1<<25; //
  int N = 1<<27;  // can't be more than 2^30 or memory errors
  int Nx = 4; // omp_get_num_threads()*2; // must be even, can be arbitrarily large
  int num_threads = 256;
  int num_blocks = 128;
  double total_percent = 0.0;
  double pi;

  float** rand_host;
  float* rand_dev1;
  float* rand_dev2;

  rand_host = (float**) malloc(Nx*sizeof(float*));
  for (int i = 0; i < Nx; i++) {
    rand_host[i] = (float*) malloc(N*sizeof(float));
  }

  CUDA_CALL(cudaMalloc(&rand_dev1, N*sizeof(float)));
  CUDA_CALL(cudaMalloc(&rand_dev2, N*sizeof(float)));


  printf("%d\n", N);

  curandGenerator_t gen;

  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  // generate floats, copy back
  for (int i = 0; i < Nx; i++) {
    CURAND_CALL(curandGenerateUniform(gen, rand_dev1, N));
    CUDA_CALL(cudaMemcpy(rand_host[i], rand_dev1, N * sizeof(float),cudaMemcpyDeviceToHost));
  }
  CURAND_CALL(curandDestroyGenerator(gen));

  cudaDeviceSynchronize();

  // make results vectors; 1 == in circle, 0 == outside circle
  int** results_host;
  int* results_dev;

  results_host = (int**) malloc(Nx/2*sizeof(int*));
  for (int i = 0; i < Nx/2; i++) {
    results_host[i] = (int*) malloc(N*sizeof(int));
  }

  CUDA_CALL(cudaMalloc(&results_dev, N*sizeof(int)));

  // make streams; one per kernel?
  int num_streams = N/num_darts;

  printf("num streams %d\n", num_streams);
  printf("making streams\n");
  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++) {
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }

  printf("starting compute\n");
  for (int n = 0; n < Nx; n +=2) {
    printf("n is %d\n", n);
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < num_streams; i++) {
      // do a bunch of async memcpys to device
      //printf("first memcpy\n");
      CUDA_CALL(cudaMemcpyAsync(&rand_dev1[i*num_darts], &rand_host[n][i*num_darts],
        num_darts*sizeof(float), cudaMemcpyHostToDevice, streams[i]));
      //printf("second memcpy\n");
      CUDA_CALL(cudaMemcpyAsync(&rand_dev2[i*num_darts], &rand_host[n+1][i*num_darts],
        num_darts*sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    }
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < num_streams; i++) {
      // launch kernels on streams
      //printf("calling kernel\n");
      montecarlo<<<num_blocks, num_threads, 0, streams[i]>>>(&rand_dev1[i*num_darts],
        &rand_dev2[i*num_darts], &results_dev[i*num_darts], num_blocks*num_threads, num_darts);
    }
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < num_streams; i++) {
      // do a bunch of async memcpys from device
      //printf("copy back\n");
      CUDA_CALL(cudaMemcpyAsync(&results_host[n/2][i*num_darts], &results_dev[i*num_darts],
        num_darts*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }
  }

  cudaDeviceSynchronize();

  // sum results

  //int num_inside[omp_get_num_threads()];
  int num_inside[4];
  #pragma omp parallel for num_threads(2)
  for (int n = 0; n < Nx/2; n++) {
    num_inside[n] = 0;
    for (int i = 0; i < N; i++) {
      #pragma omp atomic
      num_inside[n] += results_host[n][i];
    }
    printf("num darts in circle %d: %d\n", n, num_inside[n]);
  }

  double percent[Nx/2];
  #pragma omp parallel for reduction (+:total_percent) num_threads(2)
  for (int n = 0; n < Nx/2; n++) {
    percent[n] = num_inside[n] / (double)N;
    #pragma omp atomic
    total_percent += percent[n];
  }

  //total_percent /= Nx/2;
  #pragma omp single
  { pi = total_percent * 4; }

  printf("pi is %f\n", pi);


  // free things
  cudaFree(rand_dev1);
  cudaFree(rand_dev2);
  cudaFree(results_dev);

  for (int i = 0; i < Nx; i++) {
    free(rand_host[i]);
  }
  free(rand_host);

  for (int i = 0; i < Nx/2; i++) {
    free(results_host[i]);
  }
  free(results_host);
#if defined(APEX_HAVE_MPI)
  MPI_Finalize();
#endif
  apex::finalize();
  apex::cleanup();
}
