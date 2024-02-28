/*
 * Copyright (c) 2022 Kevin Huck
 * Copyright (c) 2022 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

///////////////////////////////////////////////////////////////////////////////
// Below are structures needed for tracking allocations / frees.
// These are used both on the CPU side and the GPU side.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <apex.hpp>

typedef enum apex_allocator {
    APEX_MALLOC = 0,
    APEX_CALLOC,
    APEX_REALLOC,
    APEX_GPU_HOST_MALLOC,
    APEX_GPU_DEVICE_MALLOC,
    APEX_FREE
} apex_allocator_t;

namespace apex {

void apex_report_leaks();
void apex_get_leak_symbols();

class record_t {
public:
    size_t bytes;
    task_identifier * id;
    size_t tid;
    apex_allocator_t alloc;
    record_t() : bytes(0), id(nullptr), tid(0), alloc(APEX_MALLOC), resolved(false), cpu(true) {}
    record_t(size_t b, size_t t, apex_allocator_t a, bool on_cpu) :
        bytes(b), id(nullptr), tid(t), alloc(a), resolved(false), cpu(on_cpu) {}
    //std::vector<uintptr_t> backtrace;
    std::array<void*,64> backtrace;
    std::array<std::string,64> symbols;
    bool resolved;
    size_t size;
    bool cpu;
};

class book_t {
public:
    size_t saved_node_id;
    std::atomic<size_t> totalAllocated{0};
    std::unordered_map<const void*,record_t> memoryMap;
    std::mutex mapMutex;
    ~book_t() {
        apex_report_leaks();
    }
};

book_t& getBook(void);
void controlMemoryWrapper(bool enabled);
void printBacktrace(void);
void recordAlloc(const size_t bytes, const void* ptr,
    const apex_allocator_t alloc, const bool cpu = true);
void recordFree(const void* ptr, const bool cpu = true);
void recordMetric(std::string name, double value);

}; // apex namespace

