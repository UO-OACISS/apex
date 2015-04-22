#include <string>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <execinfo.h>

static void apex_custom_signal_handler(int sig) {

  std::cerr << std::endl;
  std::cerr  << "********* " << strsignal(sig) << " *********";
  std::cerr << std::endl;
  std::cerr << std::endl;

  void *trace[32];
  size_t size, i;
  char **strings;

  size    = backtrace( trace, 32 );
  strings = backtrace_symbols( trace, size );

  std::cerr << std::endl;
  std::cerr << "BACKTRACE:";
  std::cerr << std::endl;
  std::cerr << std::endl;

  for( i = 0; i < size; i++ ){
   std::cerr << "\t" << strings[i] << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "***************************************";
  std::cerr << std::endl;
  std::cerr << std::endl;
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
