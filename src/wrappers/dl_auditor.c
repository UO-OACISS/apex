/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>

int * objopen_counter()
{
  static int count = 0;
  return &count;
}

// This auditor supports all API versions.
unsigned int la_version(unsigned int version)
{
  return version;
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  (*objopen_counter())++;
  return 0;
}

void la_preinit(uintptr_t *cookie)
{
  typedef void (*apex_memory_dl_initialized_t)();
  static apex_memory_dl_initialized_t apex_memory_dl_initialized = NULL;
  void * memory_so;

#if defined(__APPLE__)
  memory_so = dlmopen(LM_ID_BASE, "libapex_memory_wrapper.dylib", RTLD_NOW);
#else
  memory_so = dlmopen(LM_ID_BASE, "libapex_memory_wrapper.so", RTLD_NOW);
#endif

  if (memory_so) {
    char const * err;

    dlerror(); // reset error flag
    apex_memory_dl_initialized =
        (apex_memory_dl_initialized_t)dlsym(memory_so,
        "apex_memory_dl_initialized");
    // Check for errors
    if ((err = dlerror())) {
      printf("APEX: ERROR obtaining symbol info in auditor: %s\n", err);
    } else {
      apex_memory_dl_initialized();
    }
    dlclose(memory_so);
  } else {
    printf("APEX: ERROR in opening APEX library in auditor.\n");
  }
}

