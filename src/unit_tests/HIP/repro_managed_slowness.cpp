/**
* Reproduce slow kernels after hipMemcpy involving managed memory as destination
* used in a computation kernel.
*
* Defines three arrays, to simulate a common pattern in GENE:
*
* (1) managed memory array allocated from Fortran (via C interfaces to hip)
* (2) initialized from host
* (3) pass to C computation kernel which does in-place operation
* (4) copy to second managed memory array so original is not overwritten
* (5) perform some computation using the copied to managed memory array
*     and some other data (could be in device mem or managed mem)
*/
#include <iostream>
#include <time.h>

#include <hip/hip_runtime.h>
// roctx header file
#include <roctx.h>
#include "apex_api.hpp"

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }


__global__ void kernel_assign_1(const int size, double *lhs, double *rhs)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < size) {
    lhs[i] = rhs[i];
  }
}


template <typename T>
void hipMallocManaged2(T **p, size_t nbytes, int device_id)
{
  CHECK(hipMallocManaged(p, nbytes));
  CHECK(hipMemAdvise(*p, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  CHECK(hipMemAdvise(*p, nbytes, hipMemAdviseSetPreferredLocation, device_id));
  CHECK(hipMemAdvise(*p, nbytes, hipMemAdviseSetAccessedBy, device_id));
}


int main (int argc, char *argv[])
{
  const int n = 10 * 1024 * 1024;
  const int nbytes = sizeof(double) * n;
  const int block_size = 256;
  const int nwarmup = 100;
  //const double max_seconds = 2;
  double *d_a, *d_b, *d_c, *d_d;
  struct timespec start, end;
  double elapsed, total;

  int device_id = 0;
  CHECK(hipSetDevice(0));
  //CHECK(hipGetDevice(&device_id));

  // data coming in is always managed
  hipMallocManaged2(&d_a, nbytes, device_id);
  // initialize the "input" data
  for (int i = 0 ; i < n ; i++) {
    d_a[i] = i;
  }

  // intermediate managed or device to compare
#ifdef MANAGED
  hipMallocManaged2(&d_b, nbytes, device_id);
  std::string prefix{"Managed: "};
#else
  CHECK(hipMalloc(&d_b, nbytes));
  std::string prefix{"Unmanaged: "};
#endif

  // output array always in device
  CHECK(hipMalloc(&d_c, nbytes));

  // something to consume the result
  CHECK(hipHostMalloc(&d_d, nbytes, hipHostMallocNumaUser));

  total = 0.0;
  int niter = 0;
  dim3 nblocks(n / block_size);
  dim3 threads_per_block(block_size);
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  auto ci = apex::new_task("Copy Input");
  auto rk = apex::new_task("Run Kernel");
  auto co = apex::new_task("Copy output");
  //while (total < max_seconds) {
  while (niter < 300) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    apex::start(ci);
    // Copy the data from the "input" to the intermediary
    CHECK(hipMemcpy(d_b, d_a, nbytes, hipMemcpyDeviceToDevice));
    // synchronize to "time" that action
    CHECK(hipDeviceSynchronize());
    apex::stop(ci);
    apex::start(rk);
    // run the kernel, which element-wise copies from d_b to d_c
    kernel_assign_1<<<nblocks, threads_per_block, 0, stream>>>(n, d_c, d_b);
    // check for errors
    CHECK(hipGetLastError());
    // wait for the kernel to finish
    CHECK(hipStreamSynchronize(stream));
    apex::stop(rk);
    apex::start(co);
    CHECK(hipMemcpy(d_d, d_c, nbytes, hipMemcpyDeviceToHost));
    CHECK(hipDeviceSynchronize());
    apex::stop(co);
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (niter >= nwarmup) {
      elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
      total += elapsed;
    }
    niter++;
  }
  // destroy the stream
  CHECK(hipStreamDestroy(stream));
  CHECK(hipFree(d_d));
  CHECK(hipFree(d_c));
  CHECK(hipFree(d_b));
  CHECK(hipFree(d_a));
  niter -= nwarmup;
  std::cout << float(total) / niter << "\t" << niter << std::endl;

  return 0;
}

