#pragma once

#include "task_identifier.hpp"
#include "profiler.hpp"
#include <vector>

namespace apex {

struct task_wrapper {
    task_identifier * task_id;
    profiler * prof;
    uint64_t guid;
    uint64_t parent_guid;
    std::vector<profiler*> data_ptr;
    task_wrapper(void) : 
        task_id(nullptr), prof(nullptr), guid(0ull), 
        parent_guid(0ull) { }
}; // struct task_wrapper

}; // namespace apex