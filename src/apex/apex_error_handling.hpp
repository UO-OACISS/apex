#include <string>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <execinfo.h>
#include "thread_instance.hpp"
#include "address_resolution.hpp"

static void apex_custom_signal_handler(int sig) {

  fflush(stderr);
  std::cerr << std::endl;
  std::cerr << "********* Thread " << apex::thread_instance::get_id() << " " << strsignal(sig) << " *********";
  std::cerr << std::endl;
  std::cerr << std::endl;

  void *trace[32];
  size_t size, i;
  char **strings;

  size    = backtrace( trace, 32 );
  /* overwrite sigaction with caller's address */
  //trace[1] = (void *)ctx.eip;
  strings = backtrace_symbols( trace, size );

  std::cerr << std::endl;
  std::cerr << "BACKTRACE:";
  std::cerr << std::endl;
  std::cerr << std::endl;

  char exe[256];
  int len = readlink("/proc/self/exe", exe, 256);
  if (len != -1) {
    exe[len] = '\0';
  }

  // skip the first frame, it is this handler
  for( i = 1; i < size; i++ ){
   std::cerr << strings[i] << std::endl;
   //if (sig != SIGSEGV) {
     std::cerr << apex::lookup_address((uintptr_t)trace[i], false) << std::endl;
   //} else {
     //char syscom[1024];
     //sprintf(syscom,"addr2line -f -p -i -e %s %p", exe, trace[i]);
     //system(syscom);
   //}
  }

  std::cerr << std::endl;
  std::cerr << "***************************************";
  std::cerr << std::endl;
  std::cerr << std::endl;
  fflush(stderr);
  exit(-1);
}

int apex_register_signal_handler() {
  struct sigaction act;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  act.sa_handler = apex_custom_signal_handler;
  sigaction( SIGILL, &act, NULL);  
  sigaction( SIGABRT, &act, NULL);  
  sigaction( SIGFPE, &act, NULL);  
  sigaction( SIGSEGV, &act, NULL);  
  sigaction( SIGBUS, &act, NULL);  
  return 0;
}

void apex_test_signal_handler() {
  apex_custom_signal_handler(1);
}

