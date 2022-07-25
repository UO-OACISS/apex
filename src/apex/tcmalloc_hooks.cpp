/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifdef APEX_HAVE_TCMALLOC

#include "tcmalloc_hooks.hpp"
#include "gperftools/malloc_hook.h"
#include "apex_api.hpp"
#include "apex_assert.h"

namespace apex {
namespace tcmalloc {

tracker& getTracker() {
    static tracker t;
    return t;
}

bool& inWrapper() {
    thread_local static bool _inWrapper = false;
    return _inWrapper;
}

void NewHook(const void* ptr, size_t size) {
    // prevent infinite recursion...
    if (inWrapper() || apex::in_apex::get() > 0) { return; }
    inWrapper() = true;
    tracker& t = getTracker();
    double value = (double)(size);
    apex::sample_value("Memory: Bytes Allocated", value, true);
    t.hostMapMutex.lock();
    //std::cout << "Address " << ptr << " has " << size << " bytes." << std::endl;
    t.hostMemoryMap[ptr] = value;
    t.hostMapMutex.unlock();
    t.hostTotalAllocated.fetch_add(size, std::memory_order_relaxed);
    value = (double)(t.hostTotalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
    inWrapper() = false;
}

void DeleteHook(const void* ptr) {
    // prevent infinite recursion...
    if (inWrapper() || apex::in_apex::get() > 0) { return; }
    inWrapper() = true;
    tracker& t = getTracker();
    size_t size = 0;
    t.hostMapMutex.lock();
    if (t.hostMemoryMap.count(ptr) > 0) {
        size = t.hostMemoryMap[ptr];
        t.hostMemoryMap.erase(ptr);
    } else {
        //std::cerr << "Address " << ptr << " not found!" << std::endl;
        t.hostMapMutex.unlock();
        return;
    }
    t.hostMapMutex.unlock();
    double value = (double)(size);
    apex::sample_value("Memory: Bytes Freed", value, true);
    t.hostTotalAllocated.fetch_sub(size, std::memory_order_relaxed);
    value = (double)(t.hostTotalAllocated);
    apex::sample_value("Memory: Total Bytes Occupied", value);
    inWrapper() = true;
}

void init_hook() {
    if (apex_options::track_cpu_memory()) {
        getTracker();
        APEX_ASSERT(MallocHook::AddNewHook(&NewHook));
        APEX_ASSERT(MallocHook::AddDeleteHook(&DeleteHook));
    }
}

void destroy_hook() {
    if (apex_options::track_cpu_memory()) {
        APEX_ASSERT(MallocHook::RemoveNewHook(&NewHook));
        APEX_ASSERT(MallocHook::RemoveDeleteHook(&DeleteHook));
    }
}

}
}
#endif // APEX_HAVE_TCMALLOC
