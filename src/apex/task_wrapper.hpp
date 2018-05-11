#pragma once

namespace apex {
struct task_wrapper;
};

#include "task_identifier.hpp"
#include "profiler.hpp"
#include <vector>
#include <unordered_set>

namespace apex {

struct task_wrapper {
    task_identifier * task_id;
    profiler * prof;
    uint64_t guid;
    uint64_t parent_guid;
    std::shared_ptr<task_wrapper> parent;
    std::vector<profiler*> data_ptr;
    std::unordered_set<task_identifier*> aliases;
    task_wrapper(void) : 
        task_id(nullptr),
        prof(nullptr),
        guid(0ull),
        parent_guid(0ull),
        parent(nullptr)
    { }
    inline task_identifier * get_task_id(void) {
        if (!aliases.empty()) {
            task_identifier * id = nullptr;
            // find the first alias that isn't the same as the original name
            for (auto tmp : aliases) {
                if (tmp != id) {
                    id = tmp;
                    return id;
                }
            }
        }
        return task_id;
    }
    static inline std::shared_ptr<task_wrapper> & get_apex_main_wrapper(void) {
        static std::shared_ptr<task_wrapper> tt_ptr(nullptr);
        if (tt_ptr.get() != nullptr) {
            return tt_ptr;
        }
        const std::string apex_main_str("APEX MAIN");
        tt_ptr = std::make_shared<task_wrapper>();
        tt_ptr->task_id = task_identifier::get_task_id(apex_main_str);
        return tt_ptr;
    }
}; // struct task_wrapper

}; // namespace apex
