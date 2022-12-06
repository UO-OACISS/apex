#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

void* someThread(void* tmp)
{
  int* tid = (int*)tmp;
  char name[32];
  snprintf(name, 32, "worker-thread#%d", *tid);
  register_thread(name);
  profiler* p = start((apex_function_address)someThread);
  sample_value("/threadqueue{locality#0/total}/length", 2.0);
  char counter[64];
  snprintf(counter, 64, "/threadqueue{locality#0/%s}/length", name);
  sample_value(counter, 2.0);
  yield(p);
  p = start((apex_function_address)someThread);
  stop(p);
  exit_thread();
  return NULL;
}

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  init("apex::start unit test", 0, 1);
  cout << "APEX Version : " << version() << endl;
  apex_options::print_options();
  profiler* p = start(__func__);
  pthread_t thread[2];
  int tid = 0;
  pthread_create(&(thread[0]), NULL, someThread, &tid);
  int tid2 = 1;
  pthread_create(&(thread[1]), NULL, someThread, &tid2);
  pthread_join(thread[0], NULL);
  pthread_join(thread[1], NULL);
  stop(p);
  finalize();
  return 0;
}

