#include "apex_api.hpp"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

using namespace apex;
using namespace std;

profiler* swapped_timer = nullptr;

void* someThread(void* tmp)
{
  int* tid = (int*)tmp;
  char name[32];
  sprintf(name, "worker-thread#%d", *tid);
  register_thread(name);
  if (*tid == 0) {
    swapped_timer = start((apex_function_address)someThread);
    printf("Started: %p\n", (void*)swapped_timer);
  }
  sample_value("/threadqueue{locality#0/total}/length", 2.0);
  char counter[64];
  sprintf(counter, "/threadqueue{locality#0/%s}/length", name);
  sample_value(counter, 2.0);
  if (*tid == 1) {
    printf("Stopping: %p\n", (void*)swapped_timer);
    stop(swapped_timer);
  }
  exit_thread();
  return NULL;
}

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  init("apex::register_thread unit test", 0, 1);
  cout << "APEX Version : " << version() << endl;
  apex_options::untied_timers(true);
  apex_options::use_screen_output(true);
  apex_options::print_options();
  profiler* p = start("main");
  pthread_t thread[2];
  int tid = 0;
  pthread_create(&(thread[0]), NULL, someThread, &tid);
  pthread_join(thread[0], NULL);
  int tid2 = 1;
  pthread_create(&(thread[1]), NULL, someThread, &tid2);
  pthread_join(thread[1], NULL);
  stop(p);
  finalize();
  return 0;
}

