#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

extern "C" void alloc_d(long long N, float** buff);

extern "C" void alloc_d_char(long long N, char** buff);

extern "C" void free_d(char* buff);

extern "C" void init_d(long long N, char* buff, char c);

// Prints an error message containing error, function_name, file_name, line and extra_error_string.
void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string = "");

void generateG2(float* G2, int rank, size_t n_elems);

void update_local_G4(float* G2, float* G4, int rank, size_t n_elems);

void print_helper(float* G4, int index);

template<typename T>
T* allocate_on_device(std::size_t n) {
    if (n == 0)
        return nullptr;
    T* ptr;
    cudaError_t ret = cudaMalloc((void**)&ptr, n * sizeof(T));
    if (ret != cudaSuccess) {
        printErrorMessage(std::string(cudaGetErrorString(ret)), __FUNCTION__, __FILE__, __LINE__,
                          "\t DEVICE size requested : " + std::to_string(n * sizeof(T)));
        throw(std::bad_alloc());
    }
    return ptr;
}

template <typename T>
void CudaMemoryCopy(T* dest, T* src, size_t size) {
    cudaError_t ret = cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    if (ret != cudaSuccess) {
        printErrorMessage(std::string(cudaGetErrorString(ret)), __FUNCTION__, __FILE__, __LINE__, "\t cuda mem copy failed " );
        throw std::logic_error(__FUNCTION__);
    }
}