#include <string.h>
#include <stdio.h>
#include <thread>
#include "apex_api.hpp"

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

struct DataElement
{
  char *name;
  int value;
};

__global__
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
}

void launch(DataElement *elem) {
  APEX_SCOPED_TIMER;
  Kernel<<< 1, 1 >>>(elem);
  RUNTIME_API_CALL(cudaDeviceSynchronize());
}

int main(int argc, char * argv[])
{
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex::cuda unit test", 0, 1);
  apex::apex_options::use_screen_output(true);
  DataElement *e;
  RUNTIME_API_CALL(cudaMallocManaged((void**)&e, sizeof(DataElement)));
  RUNTIME_API_CALL(cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) ));
  e->value = 10;
  strcpy(e->name, "hello");

  std::vector<std::thread*> threads;

  unsigned i;
  //unsigned test_numthreads = apex::hardware_concurrency() - 1;
  unsigned test_numthreads = 3;
  for(i = 0 ; i < test_numthreads ; i++) {
    std::thread * worker = new std::thread(launch,e);
    threads.push_back(worker);
  }
  launch(e);
  for(i = 0 ; i < test_numthreads ; i++) {
    threads[i]->join();
  }

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  RUNTIME_API_CALL(cudaFree(e->name));
  RUNTIME_API_CALL(cudaFree(e));
  apex::finalize();
  apex::cleanup();
}
