#include <string.h>
#include <stdio.h>
#include "apex_api.hpp"
#if defined(APEX_HAVE_MPI)
#include "mpi.h"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

/* For user instrumentation */
#include "nvToolsExt.h"

#define ITERATIONS 4

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(_domain,name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxDomainRangePushEx(_domain, &eventAttrib); \
}
#define POP_RANGE nvtxRangePop();

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

void do_marker(const char * name) {
    nvtxMarkA(name);
}

void do_marker_payload(const char * name, int payload, nvtxDomainHandle_t domain = nullptr) {
    // zero the structure
    nvtxEventAttributes_t eventAttrib = {0};
    // set the version and the size information
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    // configure the attributes.  0 is the default for all attributes.
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF880000;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT32;
    eventAttrib.payload.iValue = payload;
    if (domain) {
        nvtxDomainMarkEx(domain, &eventAttrib);
    } else {
        nvtxMarkEx(&eventAttrib);
    }
}

void launch(DataElement *elem) {
  APEX_SCOPED_TIMER;
  nvtxDomainHandle_t domain = nvtxDomainCreateA("apex.example.loop.domain");
  do_marker_payload(elem->name, elem->value, domain);
  Kernel<<< 1, 1 >>>(elem);
  RUNTIME_API_CALL(cudaDeviceSynchronize());
  nvtxDomainDestroy(domain);
}

int main(int argc, char * argv[])
{
#if defined(APEX_HAVE_MPI)
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  apex::init("apex::cuda unit test", rank, size);
#else
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex::cuda unit test", 0, 1);
#endif
  apex::apex_options::use_screen_output(true);
  DRIVER_API_CALL(cuInit(0));
  nvtxDomainHandle_t domain = nvtxDomainCreateA("apex.example");
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = "nvtx: main"; \
  nvtxRangeId_t rid1 = nvtxDomainRangeStartEx(domain, &eventAttrib);
  printf("NVTX Range ID: %lu\n", rid1);
  do_marker("nvtx: init marker");
  nvtxRangePushA("nvtx: initialization");
  DataElement *e;
  RUNTIME_API_CALL(cudaMallocManaged((void**)&e, sizeof(DataElement)));

  e->value = 10;
  RUNTIME_API_CALL(cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) ));
  strcpy(e->name, "hello");
  nvtxRangePop();

  do_marker("nvtx: compute marker");
  PUSH_RANGE(domain, "nvtx: compute",1)
  int i;
  for(i = 0 ; i < ITERATIONS ; i++) {
    launch(e);
  }
  POP_RANGE

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  do_marker("nvtx: complete marker");
  nvtxRangeId_t rid2 = nvtxRangeStartA("nvtx: finalization");
  printf("NVTX Range ID: %lu\n", rid2);
  RUNTIME_API_CALL(cudaFree(e->name));
  RUNTIME_API_CALL(cudaFree(e));
  // intentionally interleaved
  nvtxRangeEnd(rid1);
  nvtxRangeEnd(rid2);
  nvtxDomainDestroy(domain);
#if defined(APEX_HAVE_MPI)
  MPI_Finalize();
#endif
  apex::finalize();
  apex::cleanup();
}
