#include "util_cuda.hpp"

#include <chrono>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sstream>

typedef std::chrono::high_resolution_clock Clock;

__global__
void __init_in_kernel__(long long N, char* d_array, char c)
{
  // do some computation on the device
  for(int i = 0; i<N; i++)
  {
      d_array[i] = c;
  }
}

void alloc_d(long long N, float ** buff)
{
    cudaMalloc((void**)buff, N * sizeof(float));
}

void alloc_d_char(long long N, char ** buff)
{
    cudaMalloc((void**)buff, N * sizeof(char));
}

void free_d(char* buff)
{
    cudaFree(buff);
}

void init_d(long long N, char* buff, char c)
{
    __init_in_kernel__<<<1,1>>>(N, buff, c);
}

__global__
void __generateG2_in_kernel__ (float* G2, int rank, size_t n_elems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n_elems)
    {
        G2[index] = (float)(rank + 1);
    }
}

__global__
void __update_local_G4_in_kernel__ (float* G2, float* G4, int rank, size_t n_elems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n_elems)
    {
        G4[index] += G2[index];
    }
}

void generateG2(float* G2, int rank, size_t n_elems)
{
    const int n_threads = 512;
    const int n_blocks = (n_elems + (n_threads-1))/ n_threads;
    __generateG2_in_kernel__<<<n_blocks,n_threads>>>(G2, rank, n_elems);
    return;
}

void update_local_G4(float* G2, float* G4, int rank, size_t n_elems)
{
    int n_blocks = n_elems / 512;
    __update_local_G4_in_kernel__<<<n_blocks,512>>>(G2, G4, rank, n_elems);
    return;
}

void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string) {
    std::stringstream s;

    s << "Error in function: " << function_name;
    s << " (" << file_name << ":" << line << ")" << std::endl;
    s << "The function returned: " << error << std::endl;
    if (extra_error_string != "")
        s << extra_error_string << std::endl;

    std::cout << s.str() << std::endl;
}

__global__
void __print_helper_in_kernel__ (float* G4, int index)
{
    printf("G4[%d] = %lf \n", index, G4[index]);
}

void print_helper(float* G4, int index)
{
    __print_helper_in_kernel__<<<1,1>>>(G4, index);
}