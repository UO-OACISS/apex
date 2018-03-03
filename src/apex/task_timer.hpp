#pragma once

#include "task_identifier.hpp"
#include "profiler.hpp"

namespace apex {

struct task_timer {
    task_identifier * task_id;
    profiler * prof;
    uint64_t guid;
    uint64_t parent_guid;
}; // struct task_timer

}; // namespace apex