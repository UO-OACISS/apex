/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "memory_wrapper.h"
#include <memory>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <vector>
#include "apex_api.hpp"
#include "utils.hpp"
//#include <bits/stdc++.h>

///////////////////////////////////////////////////////////////////////////////
// Below is the malloc wrapper
///////////////////////////////////////////////////////////////////////////////

typedef struct book_s {
    std::atomic<size_t> totalAllocated = 0.0;
    std::unordered_map<void*,size_t> memoryMap;
    std::mutex mapMutex;
} book_t;

book_t& getBook() {
    static book_t book;
    return book;
}

void record_alloc(size_t bytes, void* ptr) {
    static book_t& book = getBook();
    double value = (double)(bytes);
    apex::sample_value("Memory: Bytes Allocated", value, true);
    book.mapMutex.lock();
    book.memoryMap[ptr] = value;
    book.mapMutex.unlock();
    book.totalAllocated.fetch_add(bytes, std::memory_order_relaxed);
    value = (double)(book.totalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
}

void record_free(void* ptr) {
    static book_t& book = getBook();
    size_t bytes;
    book.mapMutex.lock();
    if (book.memoryMap.count(ptr) > 0) {
        bytes = book.memoryMap[ptr];
        book.memoryMap.erase(ptr);
    } else {
        book.mapMutex.unlock();
        return;
    }
    book.mapMutex.unlock();
    double value = (double)(bytes);
    apex::sample_value("Memory: Bytes Freed", value, true);
    book.totalAllocated.fetch_sub(bytes, std::memory_order_relaxed);
    value = (double)(book.totalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
}

/* We need to access this global before the memory wrapper is enabled.
 * Otherwise, when it is constructed during the first allocation, we
 * could end up with a deadlock. */
void apex_memory_wrapper_init() {
    static book_t& book = getBook();
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
    record_alloc(size, retval);
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
    record_free(ptr);
    // do the allocation
    free_call(ptr);
    inWrapper() = false;
    return;
}

// Comparator function to sort pairs descending, according to second value
bool cmp(std::pair<void*, size_t>& a,
        std::pair<void*, size_t>& b)
{
    return a.second > b.second;
}

void apex_report_leaks() {
    static book_t& book = getBook();
    // Declare vector of pairs
    std::vector<std::pair<void*, size_t> > sorted;

    // Copy key-value pair from Map
    // to vector of pairs
    book.mapMutex.lock();
    for (auto& it : book.memoryMap) {
        sorted.push_back(it);
    }
    book.mapMutex.unlock();

    // Sort using comparator function
    sort(sorted.begin(), sorted.end(), cmp);

    // Print the sorted value
    for (auto& it : sorted) {
        std::cout << it.first << " leaked " << it.second << " bytes." << std::endl;
    }
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
    record_alloc(size, retval);
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
    record_free(ptr);
    // do the allocation
    auto retval = realloc_call(ptr, size);
    // record the state
    record_alloc(size, retval);
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


