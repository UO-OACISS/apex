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

namespace apex {

void apex_report_leaks();

typedef enum allocator {
    MALLOC = 0,
    CALLOC,
    REALLOC,
    GPU_HOST_MALLOC,
    GPU_DEVICE_MALLOC
} allocator_t;

class record_t {
public:
    size_t bytes;
    task_identifier * id;
    size_t tid;
    allocator_t alloc;
    record_t() : bytes(0), id(nullptr), tid(0), alloc(MALLOC) {}
    record_t(size_t b, size_t t, allocator_t a) : bytes(b), id(nullptr), tid(t), alloc(a) {}
    //std::vector<uintptr_t> backtrace;
    std::array<void*,32> backtrace;
    size_t size;
};

class book_t {
public:
    size_t saved_node_id;
    std::atomic<size_t> totalAllocated{0};
    std::unordered_map<void*,record_t> memoryMap;
    std::mutex mapMutex;
    ~book_t() {
        apex_report_leaks();
    }
};

class backtrace_record_t {
public:
    size_t skip;
    std::vector<uintptr_t>& _stack;
    backtrace_record_t(size_t s, std::vector<uintptr_t>& _s) : skip(s), _stack(_s) {}
};

book_t& getBook(void);
void printBacktrace(void);
void recordAlloc(size_t bytes, void* ptr, allocator_t alloc, bool cpu = true);
void recordFree(void* ptr, bool cpu = false);

}; // apex namespace

