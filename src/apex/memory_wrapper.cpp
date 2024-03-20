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
#include "memory_wrapper.hpp"
#include "apex_api.hpp"
#include "apex.hpp"
#include <execinfo.h>
#include "address_resolution.hpp"
#include <stdio.h>

namespace apex {

static const char * allocator_strings[] = {
    "malloc", "calloc", "realloc", "gpu_host_malloc", "gpu_device_malloc", "free"
};

book_t& getBook() {
    static book_t book;
    return book;
}

static bool& recording(void) {
    static bool _recording{true};
    return _recording;
}

void controlMemoryWrapper(bool enabled) {
    recording() = enabled;
}

void enable_memory_wrapper() {
  if (!apex_options::track_cpu_memory()) { return; }
  typedef void (*apex_memory_initialized_t)();
  static apex_memory_initialized_t apex_memory_initialized = NULL;
  void * memory_so;

#if defined(__APPLE__)
  memory_so = dlopen("libapex_memory_wrapper.dylib", RTLD_NOW);
#else
  memory_so = dlopen("libapex_memory_wrapper.so", RTLD_NOW);
#endif

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
      printf("APEX: Starting memory tracking\n");
      apex_memory_initialized();
    }
    dlclose(memory_so);
  } else {
    printf("APEX: ERROR in opening APEX library in auditor.\n");
  }
  dlerror(); // reset error flag
}

void disable_memory_wrapper() {
  if (!apex_options::track_cpu_memory()) { return; }
  typedef void (*apex_memory_finalized_t)();
  static apex_memory_finalized_t apex_memory_finalized = NULL;
  void * memory_so;

#if defined(__APPLE__)
  memory_so = dlopen("libapex_memory_wrapper.dylib", RTLD_NOW);
#else
  memory_so = dlopen("libapex_memory_wrapper.so", RTLD_NOW);
#endif

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

void printBacktrace() {
    void *trace[64];
    size_t size, i;
    char **strings;
    size    = backtrace( trace, 64 );
    strings = backtrace_symbols( trace, size );
    std::cerr << std::endl;
    // skip the first frame, it is this handler
    for( i = 1; i < size; i++ ){
        std::cerr << demangle(strings[i]) << std::endl;
    }
}

void recordAlloc(const size_t bytes, const void* ptr,
    const apex_allocator_t alloc, const bool cpu) {
    if (!recording()) return;
    static book_t& book = getBook();
    double value = (double)(bytes);
    if (cpu) sample_value("Memory: Bytes Allocated", value, true);
    profiler * p = thread_instance::instance().get_current_profiler();
    record_t tmp(value, thread_instance::instance().get_id(), alloc, cpu);
    if (p != nullptr) { tmp.id = p->get_task_id(); }
    //backtrace_record_t rec(3,tmp.backtrace);
    //_Unwind_Backtrace (default_unwind, &(rec));
    tmp.size = backtrace(tmp.backtrace.data(), tmp.backtrace.size());
    book.mapMutex.lock();
    //book.memoryMap[ptr] = value;
    book.memoryMap.insert(std::pair<const void*,record_t>(ptr, tmp));
    book.mapMutex.unlock();
    book.totalAllocated.fetch_add(bytes, std::memory_order_relaxed);
    if (p == nullptr) {
        auto i = apex::instance();
        // might be after finalization, so double-check!
        if (i != nullptr) {
            i->the_profiler_listener->increment_main_timer_allocations(value);
        }
    } else {
        p->allocations++;
        p->bytes_allocated += value;
    }
    value = (double)(book.totalAllocated);
    if (cpu) sample_value("Memory: Total Bytes Occupied", value);
}

void recordFree(const void* ptr, const bool cpu) {
    if (!recording()) return;
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
    if (cpu) sample_value("Memory: Bytes Freed", value, true);
    book.totalAllocated.fetch_sub(bytes, std::memory_order_relaxed);
    profiler * p = thread_instance::instance().get_current_profiler();
    if (p == nullptr) {
        auto i = apex::instance();
        // might be after finalization, so double-check!
        if (i != nullptr) {
            i->the_profiler_listener->increment_main_timer_frees(value);
        }
    } else {
        p->frees++;
        p->bytes_freed += value;
    }
    value = (double)(book.totalAllocated);
    if (cpu) sample_value("Memory: Total Bytes Occupied", value);
}

/* This doesn't belong here, but whatevs */
void recordMetric(std::string name, double value) {
    in_apex prevent_memory_tracking;
    profiler * p = thread_instance::instance().get_current_profiler();
    if (p != nullptr) {
        p->metric_map[name] = value;
    }
}

// Comparator function to sort pairs descending, according to second value
bool cmp(std::pair<const void*, record_t>& a,
        std::pair<const void*, record_t>& b)
{
    return a.second.bytes > b.second.bytes;
}

// Comparator function to sort pairs descending, according to second value
bool cmp2(std::pair<std::string, size_t>& a,
        std::pair<std::string, size_t>& b)
{
    return a.second > b.second;
}

void apex_get_leak_symbols() {
    in_apex prevent_memory_tracking;
    if (!apex_options::track_cpu_memory()) { return; }
    if (!recording()) return;
    static book_t& book = getBook();
    for (auto& it : book.memoryMap) {
        for(size_t i = 0; i < it.second.size; i++ ){
            std::string* tmp2{lookup_address(((uintptr_t)it.second.backtrace[i]), true)};
            it.second.symbols[i] = *tmp2;
            //delete tmp2;
        }
        it.second.resolved = true;
    }

}

void apex_report_leaks() {
    if (!apex_options::track_gpu_memory() && !apex_options::track_cpu_memory()) {
        return;
    }
    static bool once{false};
    if (once) return;
    once = true;
    static book_t& book = getBook();
    book.saved_node_id = apex::apex::instance()->get_node_id();
    std::stringstream ss;
    ss << "memory_report." << book.saved_node_id << ".txt";
    std::string outfile{ss.str()};
    std::ofstream report (outfile);
    // Declare vector of pairs
    std::vector<std::pair<const void*, record_t> > sorted;

    if (book.saved_node_id == 0) {
        std::cout << "APEX Memory Report: (see " << outfile << ")" << std::endl;
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
        if (apex_options::use_cuda()) {
            std::cout << "Ignoring known leaks in CUDA/CUPTI..." << std::endl;
        }
        std::cout << "If there are no leaks, there won't be a file..." << std::endl;
    }
    size_t actual_leaks{0};
    // Print the sorted value
    size_t actual_bytes{0};
    for (auto& it : sorted) {
        std::stringstream ss;
        //if (it.second.bytes > 1000) {
            ss << it.second.bytes << " bytes leaked at " << std::hex << it.first << std::dec << " from task ";
        //} else {
            //break;
        //}
        std::string name{"(no timer)"};
        bool nameless{true};
        if (it.second.id != nullptr) {
            name = it.second.id->get_name();
            // skip known CUPTI leaks.
            //if (name.rfind("cuda", 0) == 0) { continue; }
            nameless = false;
        }
        ss << name << " on tid " << it.second.tid << " with backtrace: " << std::endl;
        ss << "\t" << allocator_strings[it.second.alloc] << std::endl;
        char** strings = backtrace_symbols( it.second.backtrace.data(), it.second.size );
        bool skip{false};
        for(size_t i = 3; i < it.second.size; i++ ){
            std::string tmp{strings[i]};
            if (it.second.cpu) {
                if (tmp.find("cuInit", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("libcudart", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("libcupti", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("pthread_once", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("atexit", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("apex_pthread_function", 0) != std::string::npos) { skip = true; break; }
                if (tmp.find("hipFuncGetAttributes", 0) != std::string::npos) { skip = true; break; }
                if (nameless) {
                    if (tmp.find("libcuda", 0) != std::string::npos) { skip = true; break; }
                    if (tmp.find("GOMP_parallel", 0) != std::string::npos) { skip = true; break; }
                }
            }
            const std::string unknown{"{(unknown)}"};
            if (it.second.resolved) {
                if (it.second.symbols[i].find(unknown) == std::string::npos) {
                    ss << "\t" << it.second.symbols[i] << std::endl;
                } else {
                    ss << "\t" << tmp << std::endl;
                }
            } else {
                std::string* tmp2{lookup_address(((uintptr_t)it.second.backtrace[i]), true)};
                if (tmp2->find(unknown) == std::string::npos) {
                    ss << "\t" << *tmp2 << std::endl;
                } else {
                    ss << "\t" << tmp << std::endl;
                }
            }
        }
        if (skip) { continue; }

        /*
        //for (auto a : it.second.backtrace) {
        for (size_t a = 2 ; a <  it.second.size; a++) {
            //std::string * tmp = lookup_address(a, true);
            std::string * tmp = lookup_address((uintptr_t)it.second.backtrace[a], true);
            std::string demangled = demangle(*tmp);
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
        actual_leaks++;
        actual_bytes+=it.second.bytes;
    }
    report.close();
    if (book.saved_node_id == 0) {
        std::cout << "Reported " << actual_leaks << " 'actual' leaks of "
                  << actual_bytes
                  << " bytes.\nExpect false positives if memory was freed after exit."
                  << std::endl;
    }
    if (actual_leaks == 0) {
        remove(outfile.c_str());
    }

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

} // end namespace

extern "C" void enable_memory_wrapper(void) {
    apex::enable_memory_wrapper();
}

extern "C" void disable_memory_wrapper(void) {
    apex::disable_memory_wrapper();
}

