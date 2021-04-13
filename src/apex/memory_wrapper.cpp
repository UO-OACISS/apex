/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "apex_api.hpp"

namespace apex {

void enable_memory_wrapper() {
  if (!apex_options::track_memory()) { return; }
  typedef void (*apex_memory_initialized_t)();
  static apex_memory_initialized_t apex_memory_initialized = NULL;
  void * memory_so;

  memory_so = dlopen("libapex_memory_wrapper.so", RTLD_NOW);

  if (memory_so) {
    char const * err;

    dlerror(); // reset error flag
    apex_memory_initialized =
        (apex_memory_initialized_t)dlsym(memory_so,
        "apex_memory_initialized");
    // Check for errors
    if ((err = dlerror())) {
      printf("APEX: ERROR obtaining symbol info in auditor: %s\n", err);
    } else {
      apex_memory_initialized();
      printf("APEX: Starting memory tracking\n");
    }
    dlclose(memory_so);
  } else {
    printf("APEX: ERROR in opening APEX library in auditor.\n");
  }
  dlerror(); // reset error flag
}

void disable_memory_wrapper() {
  if (!apex_options::track_memory()) { return; }
  typedef void (*apex_memory_finalized_t)();
  static apex_memory_finalized_t apex_memory_finalized = NULL;
  void * memory_so;

  memory_so = dlopen("libapex_memory_wrapper.so", RTLD_NOW);

  if (memory_so) {
    char const * err;

    dlerror(); // reset error flag
    apex_memory_finalized =
        (apex_memory_finalized_t)dlsym(memory_so,
        "apex_memory_finalized");
    // Check for errors
    if ((err = dlerror())) {
      printf("APEX: ERROR obtaining symbol info in auditor: %s\n", err);
    } else {
      apex_memory_finalized();
      //printf("APEX: Stopping memory tracking\n");
    }
    dlclose(memory_so);
  } else {
    printf("APEX: ERROR in opening APEX library in auditor.\n");
  }
  dlerror(); // reset error flag
}

}

extern "C" void enable_memory_wrapper(void) {
    apex::enable_memory_wrapper();
}

extern "C" void disable_memory_wrapper(void) {
    apex::disable_memory_wrapper();
}

