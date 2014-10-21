#include "apex.hpp"
#include <pthread.h>
#include <unistd.h>

using namespace apex;

void* someThread(void* tmp)
{
  int* tid = (int*)tmp;
  char name[32];
  sprintf(name, "worker-thread#%d", *tid);
  register_thread(name);
  void * profiler = start((void*)(someThread));
  sleep(2); // Keep it alive so we're sure the second thread gets a unique ID.
  sample_value("/threadqueue{locality#0/total}/length", 2.0);
  char counter[64];
  sprintf(counter, "/threadqueue{locality#0/%s}/length", name);
  sample_value(counter, 2.0);
  stop(profiler);
  return NULL;
}

int main (int argc, char** argv) {
  init(argc, argv, NULL);
  version();
  set_node_id(0);
  void * profiler = start((void*)(main));
  pthread_t thread[2];
  int tid = 0;
  pthread_create(&(thread[0]), NULL, someThread, &tid);
  int tid2 = 1;
  pthread_create(&(thread[1]), NULL, someThread, &tid2);
  sleep(1);
  pthread_join(thread[0], NULL);
  pthread_join(thread[1], NULL);
  stop(profiler);
  sample_value("Apex Version", version());
  finalize();
  return 0;
}

