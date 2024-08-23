/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "timer_plugin/tool_api.h"
#include <stdlib.h>
#include "apex_api.hpp"
#include "thread_instance.hpp"
#include <mutex>
#include <unordered_map>
#include <stack>

using maptype = std::unordered_map<tasktimer_guid_t,
                                   std::shared_ptr<apex::task_wrapper>>;
std::mutex mtx;

maptype& getCommonMap(void) {
    static maptype theMap;
    return theMap;
}

maptype& getMyMap(void) {
    static thread_local maptype theMap;
    return theMap;
}

void safePrint(const char * format, tasktimer_guid_t guid) {
    std::scoped_lock lock{mtx};
    printf("%lu %s GUID %lu\n", apex::thread_instance::get_id(), format, guid);
    return;
}

void safeInsert(
    tasktimer_guid_t guid,
    std::shared_ptr<apex::task_wrapper> task) {
    mtx.lock();
    getCommonMap()[guid] = task;
    mtx.unlock();
    getMyMap()[guid] = task;
    //safePrint("Inserted", guid);
}

std::shared_ptr<apex::task_wrapper> safeLookup(
    tasktimer_guid_t guid) {
    // in the thread local map?
    auto task = getMyMap().find(guid);
    if (task == getMyMap().end()) {
        // in the common map?
        std::scoped_lock lock{mtx};
        task = getCommonMap().find(guid);
        mtx.unlock();
        if (task == getCommonMap().end()) {
            safePrint("Not found", guid);
            return nullptr;
        }
        getMyMap()[guid] = task->second;
    }
    //safePrint("Found", guid);
    return task->second;
}

void safeErase(
    tasktimer_guid_t guid) {
    return;
    /*
    getMyMap().erase(guid);
    mtx.lock();
    getCommonMap().erase(guid);
    mtx.unlock();
    //safePrint("Destroyed", guid);
    */
}

extern "C" {
    // library function declarations
    void tasktimer_initialize_impl(void) {
        apex::init("PerfStubs API", 0, 1);
    }
    void tasktimer_finalize_impl(void) {
        /* Debatable whether we want to do this finalize */
        apex::finalize();
    }
    // measurement function declarations
    tasktimer_timer_t tasktimer_create_impl(
        const tasktimer_function_pointer_t function_address,
        const char* timer_name,
        const tasktimer_guid_t timer_guid,
        const tasktimer_guid_t* parent_guids,
        const uint64_t parent_count) {
        static bool& over = apex::get_program_over();
        if (over) return nullptr;
        // need to look up the parent shared pointers?
        std::vector<std::shared_ptr<apex::task_wrapper>> parent_tasks;
        for (uint64_t i = 0 ; i < parent_count ; i++) {
            auto tmp = safeLookup(parent_guids[i]);
            if (tmp != nullptr)
                parent_tasks.push_back(tmp);
        }
        // if no name, use address
        if (timer_name == nullptr || strlen(timer_name) == 0) {
            printf("Null name for timer: %p\n", function_address);
            if (parent_count > 0) {
                auto task = apex::new_task(
                                (apex_function_address)function_address,
                                timer_guid, parent_tasks);
                safeInsert(timer_guid, task);
            } else {
                auto task = apex::new_task(
                                (apex_function_address)function_address,
                                timer_guid);
                safeInsert(timer_guid, task);
            }
        } else {
            std::string tmpname{timer_name};
            //tmpname += std::string(" ");
            //tmpname += std::to_string(timer_guid);
            // TODO: need to handle multiple parents!
            if (parent_tasks.size() > 0) {
                auto task = apex::new_task(tmpname, timer_guid, parent_tasks);
                safeInsert(timer_guid, task);
            } else {
                auto task = apex::new_task(tmpname, timer_guid);
                safeInsert(timer_guid, task);
            }
        }
        return (tasktimer_timer_t)(timer_guid);
    }
    void tasktimer_schedule_impl(
        tasktimer_timer_t timer,
        tasktimer_argument_value_p arguments,
        uint64_t argument_count) {
        static bool& over = apex::get_program_over();
        if (over) return;
        // TODO: handle the schedule event, somehow
        APEX_UNUSED(timer);
        APEX_UNUSED(arguments);
        APEX_UNUSED(argument_count);
    }

#define MAP_TASK(_timer, _apex_timer) \
    static bool& over_{apex::get_program_over()}; \
    if (over_) return; \
    uint64_t _tmp = (uint64_t)(_timer); \
    auto _apex_timer = safeLookup(_tmp);

    void tasktimer_start_impl(
        tasktimer_timer_t timer,
        tasktimer_execution_space_p) {
        // TODO: capture the execution space, somehow...a new task?
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            apex::start(apex_timer);
        }
    }
    void tasktimer_yield_impl(
        tasktimer_timer_t timer) {
        static bool& over = apex::get_program_over();
        if (over) return;
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            apex::yield(apex_timer);
        }
    }
    void tasktimer_resume_impl(
        tasktimer_timer_t timer,
        tasktimer_execution_space_p) {
        // TODO: capture the execution space, somehow...a new task?
        MAP_TASK(timer, apex_timer);
        // TODO: why no resume function for task_wrapper objects?
        if (apex_timer != nullptr) {
            apex::start(apex_timer);
        }
    }
    void tasktimer_stop_impl(
        tasktimer_timer_t timer) {
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            apex::stop(apex_timer);
        }
    }
    void tasktimer_destroy_impl(
        tasktimer_timer_t timer) {
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            // TODO: need to handle the destroy event somehow.
            // definitely need to remove it from the local map.
            safeErase(apex_timer->guid);
        }
    }
    void tasktimer_add_parents_impl (
        tasktimer_timer_t timer,
        const tasktimer_guid_t* parents, const uint64_t parent_count) {
        // TODO: need to handle the add parents event
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            for (uint64_t i = 0 ; i < parent_count ; i++) {
                auto tmp = safeLookup(parents[i]);
                if (tmp != nullptr) {
                    // add the parent to the child
                    apex_timer->parents.push_back(tmp);
                }
            }
            // update the child tree
            if (apex::apex_options::use_tasktree_output() ||
                apex::apex_options::use_hatchet_output()) {
                apex_timer->assign_heritage();
            }
        }
    }
    void tasktimer_add_children_impl(
        tasktimer_timer_t timer,
        const tasktimer_guid_t* children, const uint64_t child_count) {
        // TODO: need to handle the add children event
        MAP_TASK(timer, apex_timer);
        if (apex_timer != nullptr) {
            for (uint64_t i = 0 ; i < child_count ; i++) {
                auto tmp = safeLookup(children[i]);
                if (tmp != nullptr) {
                    // add the parent to the child
                    tmp->parents.push_back(apex_timer);
                    // update the child tree
                    if (apex::apex_options::use_tasktree_output() ||
                        apex::apex_options::use_hatchet_output()) {
                        tmp->assign_heritage();
                    }
                }
            }
        }
    }

    void timerStack(
        std::shared_ptr<apex::task_wrapper> apex_timer,
        bool start) {
        static thread_local std::stack<std::shared_ptr<apex::task_wrapper>> theStack;
        if (start) {
            apex::start(apex_timer);
            theStack.push(apex_timer);
        } else {
            auto timer = theStack.top();
            apex::stop(timer);
            theStack.pop();
        }
    }

    void tasktimer_data_transfer_start_impl(
        tasktimer_guid_t guid,
        tasktimer_execution_space_p source_type,
        const char* source_name,
        const void* source_ptr,
        tasktimer_execution_space_p dest_type,
        const char* dest_name,
        const void* dest_ptr) {
        std::shared_ptr<apex::task_wrapper> parent = safeLookup(guid);
        auto task = apex::new_task("data xfer", 0, parent);
        timerStack(task, true);
    }

    void tasktimer_data_transfer_stop_impl(tasktimer_guid_t guid) {
        timerStack(nullptr, false);
    }

    void tasktimer_command_start_impl(const char* type_name) {
        // we need to create a unique GUID for the command
        static tasktimer_guid_t guid{UINT64_MAX/2};
        std::string tmpstr{type_name};
        auto task = apex::new_task(tmpstr, guid++);
        timerStack(task, true);
    }

    void tasktimer_command_stop_impl(void) {
        timerStack(nullptr, false);
    }

}

