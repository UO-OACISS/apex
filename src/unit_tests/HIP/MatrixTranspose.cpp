/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include <hip/hip_runtime.h>
#include "roctracer_ext.h"
// roctx header file
#include <roctx.h>
// openmp header file
#include <omp.h>

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    hipError_t _status = apiFuncCall;                                         \
    if (_status != hipSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, hipGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

//#define WIDTH 1024
#define WIDTH 8192
#define ITERATIONS 10

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Mark API
extern "C"
void roctracer_mark(const char* str);

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    roctxRangePush(__func__);
#pragma omp parallel for
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
    roctxRangePop();
}


int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    RUNTIME_API_CALL(hipGetDeviceProperties(&devProp, 0));

    std::cout << "Device name " << devProp.name << std::endl;
    std::cout << " System major " << devProp.major << std::endl;
    std::cout << " System minor " << devProp.minor << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // allocate the memory on the device side
    RUNTIME_API_CALL(hipMalloc((void**)&gpuMatrix, NUM * sizeof(float)));
    RUNTIME_API_CALL(hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float)));

    uint32_t iterations = ITERATIONS;
    while (iterations-- > 0) {
        int rangeId = roctxRangeStart("While Loop range");
        std::cout << "## Iteration (" << iterations << ") #################" << std::endl;

        // Memory transfer from host to device
        RUNTIME_API_CALL(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));
        RUNTIME_API_CALL(hipDeviceSynchronize());

        roctracer_mark("before HIP LaunchKernel");
        roctxMark("before hipLaunchKernel");
        roctxRangePush("hipLaunchKernel");
        // Lauching kernel from host
        hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                        gpuMatrix, WIDTH);
        roctracer_mark("after HIP LaunchKernel");
        roctxMark("after hipLaunchKernel");
        RUNTIME_API_CALL(hipDeviceSynchronize());

        // Memory transfer from device to host
        roctxRangePush("hipMemcpy");

        RUNTIME_API_CALL(hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));
        RUNTIME_API_CALL(hipDeviceSynchronize());

        roctxRangePop(); // for "hipMemcpy"
        roctxRangePop(); // for "hipLaunchKernel"

        roctxRangePush("Validation Step"); // for "validation"
        // verify the results
        errors = 0;
        double eps = 1.0E-6;
#pragma omp parallel for reduction(+:errors)
        for (i = 0; i < NUM; i++) {
            if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
                errors++;
            }
        }
        if (errors != 0) {
            printf("FAILED: %d errors\n", errors);
        } else {
            printf("PASSED!\n");
        }
        roctxRangePop(); // for "validation"
        roctxRangeStop(rangeId);
    }

    // free the resources on device side
    RUNTIME_API_CALL(hipFree(gpuMatrix));
    RUNTIME_API_CALL(hipFree(gpuTransposeMatrix));
    RUNTIME_API_CALL(hipDeviceSynchronize());

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}