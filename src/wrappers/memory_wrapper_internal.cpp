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

///////////////////////////////////////////////////////////////////////////////
// Below is the malloc wrapper
///////////////////////////////////////////////////////////////////////////////

typedef enum allocator {
    MALLOC = 0,
    CALLOC,
    REALLOC
} allocator_t;

const char * allocator_strings[] = {
    "malloc", "calloc", "realloc"
};

class record_t {
public:
    size_t bytes;
    apex::task_identifier * id;
    size_t tid;
    allocator_t alloc;
    record_t() : bytes(0), id(nullptr), tid(0), alloc(MALLOC) {}
    record_t(size_t b, size_t t, allocator_t a) : bytes(b), id(nullptr), tid(t), alloc(a) {}
    //std::vector<uintptr_t> backtrace;
    std::array<void*,32> backtrace;
    size_t size;
};

void apex_report_leaks();

extern "C" void apex_memory_lights_out();

class book_t {
public:
    size_t saved_node_id;
    std::atomic<size_t> totalAllocated = 0.0;
    std::unordered_map<void*,record_t> memoryMap;
    std::mutex mapMutex;
    ~book_t() {
        apex_memory_lights_out();
    }
};

book_t& getBook() {
    static book_t book;
    return book;
}

class backtrace_record_t {
public:
    size_t skip;
    std::vector<uintptr_t>& _stack;
    backtrace_record_t(size_t s, std::vector<uintptr_t>& _s) : skip(s), _stack(_s) {}
};

/* Unwind library callback routine.  This is passed to
   _Unwind_Backtrace.  */

/*
static _Unwind_Reason_Code
default_unwind (struct _Unwind_Context *context, void *vdata)
{
    backtrace_record_t *data = (backtrace_record_t*) vdata;
    uintptr_t pc;

#ifdef _Unwind_GetIPInfo
    int ip_before_insn = 0;
    pc = _Unwind_GetIPInfo (context, &ip_before_insn);
    if (!ip_before_insn) --pc;
#else
    pc = _Unwind_GetIP (context);
#endif

    if (data->skip > 0) {
        --data->skip;
        return _URC_NO_REASON;
    }

    data->_stack.push_back(pc);

    return _URC_NO_REASON;
}
*/

inline void printBacktrace() {
  void *trace[32];
  size_t size, i;
  char **strings;
  size    = backtrace( trace, 32 );
  strings = backtrace_symbols( trace, size );
  std::cerr << std::endl;
  // skip the first frame, it is this handler
  for( i = 1; i < size; i++ ){
   std::cerr << apex::demangle(strings[i]) << std::endl;
  }
}

void record_alloc(size_t bytes, void* ptr, allocator_t alloc) {
    static book_t& book = getBook();
    double value = (double)(bytes);
    apex::sample_value("Memory: Bytes Allocated", value, true);
    apex::profiler * p = apex::thread_instance::instance().get_current_profiler();
    record_t tmp(value, apex::thread_instance::instance().get_id(), alloc);
    if (p != nullptr) { tmp.id = p->get_task_id(); }
    //backtrace_record_t rec(3,tmp.backtrace);
    //_Unwind_Backtrace (default_unwind, &(rec));
    tmp.size = backtrace(tmp.backtrace.data(), tmp.backtrace.size());
    book.mapMutex.lock();
    //book.memoryMap[ptr] = value;
    book.memoryMap.insert(std::pair<void*,record_t>(ptr, tmp));
    book.mapMutex.unlock();
    book.totalAllocated.fetch_add(bytes, std::memory_order_relaxed);
    value = (double)(book.totalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
    if (p == nullptr) {
        auto i = apex::apex::instance();
        // might be after finalization, so double-check!
        if (i != nullptr) {
            i->the_profiler_listener->increment_main_timer_allocations(value);
        }
    } else {
        p->allocations++;
        p->bytes_allocated += value;
    }
}

void record_free(void* ptr) {
    static book_t& book = getBook();
    size_t bytes;
    book.mapMutex.lock();
    if (book.memoryMap.count(ptr) > 0) {
        record_t& tmp = book.memoryMap[ptr];
        bytes = tmp.bytes;
        book.memoryMap.erase(ptr);
    } else {
        //std::cout << std::hex << ptr << std::dec << " NOT FOUND" << std::endl;
        //printBacktrace();
        book.mapMutex.unlock();
        return;
    }
    book.mapMutex.unlock();
    double value = (double)(bytes);
    apex::sample_value("Memory: Bytes Freed", value, true);
    book.totalAllocated.fetch_sub(bytes, std::memory_order_relaxed);
    value = (double)(book.totalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
    apex::profiler * p = apex::thread_instance::instance().get_current_profiler();
    if (p == nullptr) {
        auto i = apex::apex::instance();
        // might be after finalization, so double-check!
        if (i != nullptr) {
            i->the_profiler_listener->increment_main_timer_frees(value);
        }
    } else {
        p->frees++;
        p->bytes_freed += value;
    }
}

/* We need to access this global before the memory wrapper is enabled.
 * Otherwise, when it is constructed during the first allocation, we
 * could end up with a deadlock. */
void apex_memory_wrapper_init() {
    static book_t& book = getBook();
    apex::apex_options::track_memory(true);
    getBook().saved_node_id = apex::apex::instance()->get_node_id();
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
    record_alloc(size, retval, MALLOC);
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
    if (ptr != nullptr) { record_free(ptr); }
    // do the allocation
    free_call(ptr);
    inWrapper() = false;
    return;
}

// Comparator function to sort pairs descending, according to second value
bool cmp(std::pair<void*, record_t>& a,
        std::pair<void*, record_t>& b)
{
    return a.second.bytes > b.second.bytes;
}

// Comparator function to sort pairs descending, according to second value
bool cmp2(std::pair<std::string, size_t>& a,
        std::pair<std::string, size_t>& b)
{
    return a.second > b.second;
}

void apex_report_leaks() {
    static book_t& book = getBook();
    std::stringstream ss;
    ss << "memory_report." << book.saved_node_id << ".txt";
    std::string tmp{ss.str()};
    std::ofstream report (tmp);
    // Declare vector of pairs
    std::vector<std::pair<void*, record_t> > sorted;

    if (book.saved_node_id == 0) {
        std::cout << "APEX Memory Report:" << std::endl;
        std::cout << "sorting " << book.memoryMap.size() << " leaks by size..." << std::endl;
    }

    // Copy key-value pair from Map
    // to vector of pairs
    for (auto& it : book.memoryMap) {
        sorted.push_back(it);
    }

    // Sort using comparator function
    sort(sorted.begin(), sorted.end(), cmp);

    //std::unordered_map<std::string, size_t> locations;

    if (book.saved_node_id == 0) {
        std::cout << "Aggregating leaks by task and writing report..." << std::endl;
#ifdef APEX_WITH_CUDA
        std::cout << "Ignoring known leaks in CUPTI..." << std::endl;
#endif
    }
    // Print the sorted value
    for (auto& it : sorted) {
        std::stringstream ss;
        //if (it.second.bytes > 1000) {
            ss << it.second.bytes << " bytes leaked at " << std::hex << it.first << std::dec << " from task ";
        //} else {
            //break;
        //}
        std::string name{"(no timer)"};
        if (it.second.id != nullptr) {
            name = it.second.id->name;
            // skip known CUPTI leaks.
            if (name.rfind("cuda", 0) == 0) { continue; }
        }
        ss << name << " on tid " << it.second.tid << " with backtrace: " << std::endl;
        ss << "\t" << allocator_strings[it.second.alloc] << std::endl;
        char** strings = backtrace_symbols( it.second.backtrace.data(), it.second.size );
        for(size_t i = 3; i < it.second.size; i++ ){
            ss << "\t" << apex::demangle(strings[i]) << std::endl;
        }

        /*
        //for (auto a : it.second.backtrace) {
        for (size_t a = 2 ; a <  it.second.size; a++) {
            //std::string * tmp = apex::lookup_address(a, true);
            std::string * tmp = apex::lookup_address((uintptr_t)it.second.backtrace[a], true);
            std::string demangled = apex::demangle(*tmp);
            ss << "\t" << demangled << std::endl;
        }
        */
        ss << std::endl;
        /*
        if (locations.count(name) > 0) {
            locations[name] += it.second.bytes;
        } else {
            locations[name] = it.second.bytes;
        }
        */
        report << ss.str();
    }
    report.close();

/*
    std::cout << "sorting task leaks by size..." << std::endl;
    // Declare vector of pairs
    std::vector<std::pair<std::string, size_t> > sorted2;

    // Copy key-value pair from Map to vector of pairs
    for (auto& it : locations) {
        sorted2.push_back(it);
    }

    // Sort using comparator function
    sort(sorted2.begin(), sorted2.end(), cmp2);

    // print the locations
    for (auto& l : sorted2) {
        std::cout << l.first << " leaked " << l.second << " bytes." << std::endl;
    }
*/
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
    record_alloc(size, retval, CALLOC);
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
    if (ptr != nullptr) { record_free(ptr); }
    // do the allocation
    auto retval = realloc_call(ptr, size);
    // record the state
    record_alloc(size, retval, REALLOC);
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


