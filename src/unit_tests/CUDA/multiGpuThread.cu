#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
#include <unistd.h>
#include "apex_api.hpp"

#define ARR_SIZE    10
#define NUM_THR  4

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

typedef struct {
    int *arr;
    int *dev_arr;
    int *dev_result;
    int *result;
    int dev_num;
    int thr_num;
} cuda_st;

__global__ void kernel_fc(int *dev_arr, int *dev_result)
{
    int idx = threadIdx.x;
    //printf("dev_arr[%d] = %d\n", idx, dev_arr[idx]);
    atomicAdd(dev_result, dev_arr[idx]);
}

void *thread_func(void* struc)
{
    APEX_SCOPED_TIMER;
    cuda_st * data = (cuda_st*)struc;
    printf("thread %d func start\n", data->thr_num);
    int i;
    /*
    printf("arr %d = ", data->dev_num);
    for(i=0; i<10; i++) {
        printf("%d ", data->arr[i]);
    }
    printf("\n");
    */
    RUNTIME_API_CALL(cudaSetDevice(data->dev_num));
    cudaStream_t stream;
    RUNTIME_API_CALL(cudaStreamCreate(&stream));
    for (i=0 ; i<10 ; i++) {
        RUNTIME_API_CALL(cudaMemcpy(data->dev_arr, data->arr,
            sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice));
        kernel_fc<<<1,ARR_SIZE,0,stream>>>(data->dev_arr, data->dev_result);
        RUNTIME_API_CALL(cudaMemcpy(data->result, data->dev_result,
            sizeof(int), cudaMemcpyDeviceToHost));
        RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    }
    RUNTIME_API_CALL(cudaStreamDestroy(stream));
    printf("thread %d func exit\n", data->thr_num);
    return NULL;
}

int main(void)
{
    apex::init("apex::cuda multiple thread test", 0, 1);
    apex::apex_options::use_screen_output(true);

	int count = 0;
	RUNTIME_API_CALL(cudaGetDeviceCount(&count));
	printf("%d devices found.\n", count);

    for(int i=0; i<count; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));
        cudaDeviceProp deviceProp;
        RUNTIME_API_CALL(cudaGetDeviceProperties(&deviceProp, i));
        std::cout << "Using device " << i << ", name: " << deviceProp.name << std::endl;
    }

// Make object
    cuda_st cuda[count][NUM_THR];

    // Make thread
    pthread_t pthread[count*NUM_THR];

    // Host array memory allocation
    int *arr[count];
    for(int i=0; i<count; i++) {
        arr[i] = (int*)malloc(sizeof(int)*ARR_SIZE);
    }

    // Fill this host array up with specified data
    for(int i=0; i<count; i++) {
        for(int j=0; j<ARR_SIZE; j++) {
            arr[i][j] = i*ARR_SIZE+j;
        }
    }

    // To confirm host array data
    /*
    for(int i=0; i<count; i++) {
        printf("arr[%d] = ", i);
        for(int j=0; j<ARR_SIZE; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
    */

    // Result memory allocation
    int *result[count];
    for(int i=0; i<count; i++) {
        result[i] = (int*)malloc(sizeof(int));
        memset(result[i], 0, sizeof(int));
    }

    // Device array memory allocation
    int *dev_arr[count];
    for(int i=0; i<count; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));
        RUNTIME_API_CALL(cudaMalloc(&dev_arr[i], sizeof(int)*ARR_SIZE));
    }

    // Device result memory allocation
    int *dev_result[count];
    for(int i=0; i<count; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));
        RUNTIME_API_CALL(cudaMalloc(&dev_result[i], sizeof(int)));
        RUNTIME_API_CALL(cudaMemset(dev_result[i], 0, sizeof(int)));
    }

    // Connect these pointers with object
    for (int i=0; i<count; i++) {
        for (int j=0; j<NUM_THR; j++) {
            cuda[i][j].arr = arr[i];
            cuda[i][j].dev_arr = dev_arr[i];
            cuda[i][j].result = result[i];
            cuda[i][j].dev_result = dev_result[i];
            cuda[i][j].dev_num = i;
            cuda[i][j].thr_num = j;
        }
    }

    // Create and excute pthread
    for(int i=0; i<count; i++) {
        for (int j=0; j<NUM_THR; j++) {
            pthread_create(&pthread[(i*NUM_THR)+j], NULL, thread_func, (void*)&cuda[i][j]);
        }

        // Join pthread
        for(int j=0; j<NUM_THR; j++) {
            pthread_join(pthread[j], NULL);
            //printf("result[%d][%d] = %d\n", i,j, (*cuda[i][j].result));
        }
    }

    for(int i=0; i<count; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));
        RUNTIME_API_CALL(cudaFree(dev_arr[i]));
    }
    for(int i=0; i<count; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));
        RUNTIME_API_CALL(cudaFree(dev_result[i]));
    }

    apex::finalize();
    apex::cleanup();
    return 0;
}

