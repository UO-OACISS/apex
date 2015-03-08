#include "apex.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

void* someThread(void* tmp)
{
  int* tid = (int*)tmp;
  char name[32];
  sprintf(name, "worker-thread#%d", *tid);
  register_thread(name);
  profiler* p = start((apex_function_address)someThread);
  sample_value("/threadqueue{locality#0/total}/length", 2.0);
  char counter[64];
  sprintf(counter, "/threadqueue{locality#0/%s}/length", name);
  sample_value(counter, 2.0);
  stop(p);
  return NULL;
}

int main (int argc, char** argv) {
  init(argc, argv, NULL);
  cout << "APEX version : " << version() << endl;
  set_node_id(0);
  profiler* p = start((apex_function_address)(main));
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

