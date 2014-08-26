#include "ittnotify.h"
#include "legacy/ittnotify.h"
#include <unistd.h>

int main (int argc, char** argv) {
  __itt_domain* domain = __itt_domain_create("Example.Domain.Global");
  __itt_string_handle* handle_main = __itt_string_handle_create("ittTest.main");
  __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
  sleep(1);
  __itt_task_end(domain);
  return 0;
}

