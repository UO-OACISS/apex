//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

#include <string>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <execinfo.h>
#include "thread_instance.hpp"
#include "address_resolution.hpp"
#include <errno.h>
#include <string.h>
#include <regex>
#include "utils.hpp"

static std::mutex output_mutex;

static void apex_custom_signal_handler(int sig) {

  int errnum = errno;

  //std::unique_lock<std::mutex> l(output_mutex);
  fflush(stderr);
  std::cerr << std::endl;
  std::cerr << "********* Thread " << apex::thread_instance::get_id() << " " <<
  strsignal(sig) << " *********";
  std::cerr << std::endl;
  std::cerr << std::endl;
  if(errnum) {
    std::cerr << "Value of errno: " << errno << std::endl;
    perror("Error printed by perror");
    std::cerr << "Error string: " << strerror( errnum ) << std::endl;
  }

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
   std::cerr << apex::demangle(strings[i]) << std::endl;
// #if APEX_HAVE_BFD
   // std::cerr << apex::lookup_address((uintptr_t)trace[i], false) << std::endl;
// #else
   char syscom[1024];
#ifndef __APPLE__
   sprintf(syscom,"addr2line -f -i -e %s %p", exe, trace[i]);
#endif
   system(syscom);
// #endif
  }

  std::cerr << std::endl;
  std::cerr << "***************************************";
  std::cerr << std::endl;
  std::cerr << std::endl;
  fflush(stderr);
  //apex::finalize();
  _exit(-1);
}

int apex_register_signal_handler() {
  struct sigaction act;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  act.sa_handler = apex_custom_signal_handler;
  sigaction( SIGILL, &act, nullptr);
  sigaction( SIGABRT, &act, nullptr);
  sigaction( SIGFPE, &act, nullptr);
  sigaction( SIGSEGV, &act, nullptr);
  sigaction( SIGBUS, &act, nullptr);
  return 0;
}

void apex_test_signal_handler() {
  apex_custom_signal_handler(1);
}

