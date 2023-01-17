/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "memory_wrapper.h"
#include "memory_wrapper.hpp"
#include <memory>
#include <atomic>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include "apex_api.hpp"
#include "thread_instance.hpp"
#include "address_resolution.hpp"
#include "utils.hpp"
//#include <bits/stdc++.h>
//for unwinding
#include <unwind.h>
#include <stdint.h>
// for backtrace
#include <execinfo.h>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////
// Below is the malloc wrapper
///////////////////////////////////////////////////////////////////////////////

/* We need to access this global before the memory wrapper is enabled.
 * Otherwise, when it is constructed during the first allocation, we
 * could end up with a deadlock. */
void apex_memory_wrapper_init() {
    static apex::book_t& book = apex::getBook();
    apex::apex_options::track_cpu_memory(true);
    apex::getBook().saved_node_id = apex::apex::instance()->get_node_id();
    atexit(apex_memory_lights_out);
    APEX_UNUSED(book);
}

bool& inWrapper() {
    thread_local static bool _inWrapper = false;
    return _inWrapper;
}

void* apex_malloc_wrapper(malloc_p malloc_call, size_t size) {
    if(inWrapper() || apex::in_apex::get() > 0) {
        // Another wrapper has already intercepted the call so just pass through
        return malloc_call(size);
    }
    inWrapper() = true;
    // do the allocation
    auto retval = malloc_call(size);
    // record the state
    apex::recordAlloc(size, retval, apex::MALLOC);
    inWrapper() = false;
    return retval;
}

void apex_free_wrapper(free_p free_call, void* ptr) {
    if(inWrapper() || apex::in_apex::get() > 0) {
        // Another wrapper has already intercepted the call so just pass through
        return free_call(ptr);
    }
    inWrapper() = true;
    // record the state
    if (ptr != nullptr) { apex::recordFree(ptr); }
    // do the allocation
    free_call(ptr);
    inWrapper() = false;
    return;
}

void* apex_calloc_wrapper(calloc_p calloc_call, size_t nmemb, size_t size) {
    if(inWrapper() || apex::in_apex::get() > 0) {
        // Another wrapper has already intercepted the call so just pass through
        return calloc_call(nmemb, size);
    }
    inWrapper() = true;
    // do the allocation
    auto retval = calloc_call(nmemb, size);
    // record the state
    apex::recordAlloc(size, retval, apex::CALLOC);
    inWrapper() = false;
    return retval;
}

void* apex_realloc_wrapper(realloc_p realloc_call, void* ptr, size_t size) {
    if(inWrapper() || apex::in_apex::get() > 0) {
        // Another wrapper has already intercepted the call so just pass through
        return realloc_call(ptr, size);
    }
    inWrapper() = true;
    // record the state
    if (ptr != nullptr) { apex::recordFree(ptr); }
    // do the allocation
    auto retval = realloc_call(ptr, size);
    // record the state
    apex::recordAlloc(size, retval, apex::REALLOC);
    inWrapper() = false;
    return retval;
}

#if 0
#if defined(memalign)
extern "C"
void* apex_memalign_wrapper(memalign_p memalign_call, size_t nmemb, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return memalign_call(nmemb, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = memalign_call(nmemb, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(reallocarray)
extern "C"
void* apex_reallocarray_wrapper(reallocarray_p reallocarray_call, void* ptr, size_t nmemb, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return reallocarray_call(ptr, nmemb, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = reallocarray_call(ptr, nmemb, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(reallocf)
extern "C"
void* apex_reallocf_wrapper(reallocf_p reallocf_call, void* ptr, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return reallocf_call(ptr, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = reallocf_call(ptr, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(valloc)
extern "C"
void* apex_valloc_wrapper(valloc_p valloc_call, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return valloc_call(size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = valloc_call(size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(malloc_usable_size)
extern "C"
size_t apex_malloc_usable_size_wrapper(malloc_usable_size_p malloc_usable_size_call, void* ptr) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return malloc_usable_size_call(ptr);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = malloc_usable_size_call(ptr);

        inWrapper() = false;
        return retval;
    }
}
#endif

#endif

extern "C" void* apex_malloc(size_t size) {
  return apex_malloc_wrapper(malloc, size);
}

extern "C" void apex_free(void* ptr) {
  return apex_free_wrapper(free, ptr);
}

extern "C" void* apex_calloc(size_t nmemb, size_t size) {
  return apex_calloc_wrapper(calloc, nmemb, size);
}

extern "C" void* apex_realloc(void* ptr, size_t size) {
  return apex_realloc_wrapper(realloc, ptr, size);
}

#if 0
#if defined(memalign)
extern "C" void* apex_memalign(size_t nmemb, size_t size) {
  return apex_memalign_wrapper(memalign, nmemb, size);
}
#endif

#if defined(reallocarray)
extern "C" void* apex_reallocarray(void* ptr, size_t nmemb, size_t size) {
  return apex_reallocarray_wrapper(reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
extern "C" void* apex_reallocf(void* ptr, size_t size) {
  return apex_reallocf_wrapper(reallocf, ptr, size);
}
#endif

#if defined(valloc)
extern "C" void* apex_valloc(size_t size) {
  return apex_valloc_wrapper(valloc, size);
}
#endif

#if defined(malloc_usable_size)
extern "C" void* apex_malloc_usable_size(void* ptr) {
  return apex_malloc_usable_size_wrapper(malloc_usable_size, ptr);
}
#endif

#endif


