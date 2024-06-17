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

void safeInsert(
    tasktimer_guid_t guid,
    std::shared_ptr<apex::task_wrapper> task) {
    mtx.lock();
    getCommonMap()[guid] = task;
    mtx.unlock();
    getMyMap()[guid] = task;
}

std::shared_ptr<apex::task_wrapper> safeLookup(
    tasktimer_guid_t guid) {
    // in the thread local map?
    auto task = getMyMap().find(guid);
    if (task == getMyMap().end()) {
        // in the common map?
        mtx.lock();
        task = getCommonMap().find(guid);
        mtx.unlock();
        getMyMap()[guid] = task->second;
    }
    return task->second;
}

void safeErase(
    tasktimer_guid_t guid) {
    getMyMap().erase(guid);
    mtx.lock();
    getCommonMap().erase(guid);
    mtx.unlock();
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
        // TODO: need to handle multiple parents!
        // need to look up the parent shared pointers?
        std::vector<std::shared_ptr<apex::task_wrapper>> parent_tasks;
        for (uint64_t i = 0 ; i < parent_count ; i++) {
            parent_tasks.push_back(safeLookup(parent_guids[i]));
        }
        // if no name, use address
        if (timer_name == nullptr || strlen(timer_name) == 0) {
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
            if (parent_count > 0) {
                auto task = apex::new_task(timer_name, timer_guid, parent_tasks);
                safeInsert(timer_guid, task);
            } else {
                auto task = apex::new_task(timer_name, timer_guid);
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
        apex::start(apex_timer);
    }
    void tasktimer_yield_impl(
        tasktimer_timer_t timer) {
        static bool& over = apex::get_program_over();
        if (over) return;
        MAP_TASK(timer, apex_timer);
        apex::yield(apex_timer);
    }
    void tasktimer_resume_impl(
        tasktimer_timer_t timer,
        tasktimer_execution_space_p) {
        // TODO: capture the execution space, somehow...a new task?
        MAP_TASK(timer, apex_timer);
        // TODO: why no resume function for task_wrapper objects?
        apex::start(apex_timer);
    }
    void tasktimer_stop_impl(
        tasktimer_timer_t timer) {
        MAP_TASK(timer, apex_timer);
        apex::stop(apex_timer);
    }
    void tasktimer_destroy_impl(
        tasktimer_timer_t timer) {
        MAP_TASK(timer, apex_timer);
        // TODO: need to handle the destroy event somehow.
        // definitely need to remove it from the local map.
        safeErase(apex_timer->guid);
    }
    void tasktimer_add_parents_impl (
        tasktimer_timer_t timer,
        const tasktimer_guid_t* parents, const uint64_t parent_count) {
        // TODO: need to handle the add parents event
        MAP_TASK(timer, apex_timer);
        APEX_UNUSED(apex_timer);
        APEX_UNUSED(parents);
        APEX_UNUSED(parent_count);
    }
    void tasktimer_add_children_impl(
        tasktimer_timer_t timer,
        const tasktimer_guid_t* children, const uint64_t child_count) {
        // TODO: need to handle the add children event
        MAP_TASK(timer, apex_timer);
        APEX_UNUSED(apex_timer);
        APEX_UNUSED(children);
        APEX_UNUSED(child_count);
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
        std::string tmpstr{type_name};
        auto task = apex::new_task(tmpstr);
        timerStack(task, true);
    }

    void tasktimer_command_stop_impl(void) {
        timerStack(nullptr, false);
    }

}

