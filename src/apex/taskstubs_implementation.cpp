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
#include <stdarg.h>


using maptype = std::unordered_map<tasktimer_guid_t,
                                   std::shared_ptr<apex::task_wrapper>>;

std::mutex& mtx(void) {
    static std::mutex mtx;
    return mtx;
}

maptype& getCommonMap(void) {
    static maptype theMap;
    return theMap;
}

maptype& getMyMap(void) {
    static thread_local maptype theMap;
    return theMap;
}

void verbosePrint(const char *format, ...)
{
    static std::mutex local_mtx;
    std::scoped_lock lock{local_mtx};
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

#define VERBOSE_PRINTF(...) if (apex::apex_options::use_verbose()) { verbosePrint(__VA_ARGS__); }

void safePrint(const char * format, tasktimer_guid_t guid, const char * event) {
    //static std::mutex local_mtx;
    //std::scoped_lock lock{local_mtx};
    VERBOSE_PRINTF("%lu TS: %s: GUID %p %s\n", apex::thread_instance::get_id(), format, guid, event);
    return;
}

void safeInsert(
    tasktimer_guid_t guid,
    std::shared_ptr<apex::task_wrapper> task) {
#if 0
    {
        std::scoped_lock lock{mtx()};
        getCommonMap()[guid] = task;
    }
    getMyMap()[guid] = task;
#else
    std::scoped_lock lock{mtx()};
    getCommonMap()[guid] = task;
#endif
    //safePrint("Inserted", guid);
}

std::shared_ptr<apex::task_wrapper> safeLookup(
    tasktimer_guid_t guid, const char * event) {
#if 0
    // in the thread local map?
    auto task = getMyMap().find(guid);
    if (task == getMyMap().end()) {
        // in the common map?
        {
            std::scoped_lock lock{mtx()};
            task = getCommonMap().find(guid);
        }
        if (task == getCommonMap().end()) {
            safePrint("Not found", guid);
            APEX_ASSERT(false);
            return nullptr;
        }
        getMyMap()[guid] = task->second;
    }
#else
    std::scoped_lock lock{mtx()};
    auto task = getCommonMap().find(guid);
    if (task == getCommonMap().end()) {
        safePrint("Not found", guid, event);
        //APEX_ASSERT(false);
        return nullptr;
    }
 #endif
    //safePrint("Found", guid);
    return task->second;
}

void safeErase(
    tasktimer_guid_t guid) {
    APEX_UNUSED(guid);
    return;
    /*
    getMyMap().erase(guid);
    */
    {
        std::scoped_lock lock{mtx()};
        getCommonMap().erase(guid);
    }
    //safePrint("Destroyed", guid);
    return;
}

extern "C" {
    // library function declarations
    void tasktimer_initialize_impl(void) {
        apex::init("PerfStubs API", 0, 1);
        apex::apex_options::use_thread_flow(true);
    }
    void tasktimer_finalize_impl(void) {
        /* Debatable whether we want to do this finalize */
        //apex::finalize();
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
        safePrint("Creating", timer_guid, timer_name);

        // need to look up the parent shared pointers?
        std::vector<std::shared_ptr<apex::task_wrapper>> parent_tasks;
        for (uint64_t i = 0 ; i < parent_count ; i++) {
            auto tmp = safeLookup(parent_guids[i], "parent lookup");
            if (tmp != nullptr)
                parent_tasks.push_back(tmp);
        }
        // if no name, use address
        if (timer_name == nullptr || strlen(timer_name) == 0) {
            //VERBOSE_PRINTF("Null name for timer: %p\n", function_address);
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

#define MAP_TASK(_timer, _apex_timer, _event) \
    static bool& over_{apex::get_program_over()}; \
    if (over_) return; \
    uint64_t _tmp = (uint64_t)(_timer); \
    auto _apex_timer = safeLookup(_tmp, _event);

    void tasktimer_schedule_impl(
        tasktimer_timer_t timer,
        tasktimer_argument_value_p arguments,
        uint64_t argument_count) {
        MAP_TASK(timer, apex_timer, "schedule");
        VERBOSE_PRINTF("%lu TS: Scheduling: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        static bool& over = apex::get_program_over();
        if (over) return;
        for (uint64_t i = 0 ; i < argument_count ; i++) {
            switch (arguments[i].type) {
                case TASKTIMER_LONG_INTEGER_TYPE: {
                    apex::task_wrapper::argument tmp = (int64_t)arguments[i].l_value;
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_LONG_INTEGER_TYPE);
                    break;
                }
                case TASKTIMER_UNSIGNED_LONG_INTEGER_TYPE: {
                    apex::task_wrapper::argument tmp = (uint64_t)arguments[i].u_value;
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_UNSIGNED_LONG_INTEGER_TYPE);
                    break;
                }
                case TASKTIMER_DOUBLE_TYPE: {
                    apex::task_wrapper::argument tmp = arguments[i].d_value;
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_DOUBLE_TYPE);
                    break;
                }
                case TASKTIMER_STRING_TYPE: {
                    apex::task_wrapper::argument tmp = std::string(arguments[i].c_value);
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_STRING_TYPE);
                    break;
                }
                case TASKTIMER_POINTER_TYPE: {
                    apex::task_wrapper::argument tmp = arguments[i].p_value;
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_POINTER_TYPE);
                    break;
                }
                case TASKTIMER_ARRAY_TYPE: {
                    apex::task_wrapper::argument tmp = arguments[i].a_value;
                    apex_timer->arguments.push_back(tmp);
                    apex_timer->argument_types.push_back(APEX_ARRAY_TYPE);
                    break;
                }
                default:
                    break;
            }
            if (arguments[i].name != nullptr) {
                apex_timer->argument_names.push_back(std::string(arguments[i].name));
            } else {
                std::string tmpname{"arg["};
                tmpname = tmpname + std::to_string(i+2); // add two, because GUID and parent GUID are 0,1
                tmpname = tmpname + "]";
                apex_timer->argument_names.push_back(tmpname);
            }
        }
    }

    void tasktimer_start_impl(
        tasktimer_timer_t timer,
        tasktimer_execution_space_p) {
        // TODO: capture the execution space, somehow...a new task?
        MAP_TASK(timer, apex_timer, "start");
        VERBOSE_PRINTF("%lu TS: Starting: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            apex::start(apex_timer);
        }
    }
    void tasktimer_yield_impl(
        tasktimer_timer_t timer) {
#if 1
        static bool& over = apex::get_program_over();
        if (over) return;
        MAP_TASK(timer, apex_timer, "yield");
        VERBOSE_PRINTF("%lu TS: Yielding: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            apex::yield(apex_timer);
        }
#endif
    }
    void tasktimer_resume_impl(
        tasktimer_timer_t timer,
        tasktimer_execution_space_p) {
#if 1
        // TODO: capture the execution space, somehow...a new task?
        MAP_TASK(timer, apex_timer, "resume");
        VERBOSE_PRINTF("%lu TS: Resuming: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        // TODO: why no resume function for task_wrapper objects?
        if (apex_timer != nullptr) {
            apex::resume(apex_timer);
        } else {
            APEX_ASSERT(false);
        }
#endif
    }
    void tasktimer_stop_impl(
        tasktimer_timer_t timer) {
        static std::set<tasktimer_timer_t> stopped;
        static std::mutex local_mtx;
        MAP_TASK(timer, apex_timer, "stop");
        VERBOSE_PRINTF("%lu TS: Stopping: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            {
                std::scoped_lock lock{local_mtx};
                if (stopped.count(timer) > 0) {
                    VERBOSE_PRINTF("%lu TS: ERROR! TIMER STOPPED TWICE! : %p\n", apex::thread_instance::get_id(), timer);
                    return;
                }
                stopped.insert(timer);
            }
            apex::stop(apex_timer);
        }
    }
    void tasktimer_destroy_impl(
        tasktimer_timer_t timer) {
        MAP_TASK(timer, apex_timer, "destroy");
        VERBOSE_PRINTF("%lu TS: Destroying: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            //if (apex_timer->state == apex::task_wrapper::RUNNING) { apex::stop(apex_timer); }
            // TODO: need to handle the destroy event somehow.
            // definitely need to remove it from the local map.
            safeErase(apex_timer->guid);
        }
    }
    void tasktimer_add_parents_impl (
        tasktimer_timer_t timer,
        const tasktimer_guid_t* parents, const uint64_t parent_count) {
        // TODO: need to handle the add parents event
        MAP_TASK(timer, apex_timer, "add parents");
        VERBOSE_PRINTF("%lu TS: Adding parents: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            for (uint64_t i = 0 ; i < parent_count ; i++) {
                auto tmp = safeLookup(parents[i], "parent lookup");
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
        MAP_TASK(timer, apex_timer, "add children");
        VERBOSE_PRINTF("%lu TS: Adding children: %p %s\n", apex::thread_instance::get_id(), timer,
            apex_timer == nullptr ? "unknown" : apex_timer->task_id->get_name().c_str());
        if (apex_timer != nullptr) {
            for (uint64_t i = 0 ; i < child_count ; i++) {
                auto tmp = safeLookup(children[i], "child lookup");
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
        std::shared_ptr<apex::task_wrapper> parent = safeLookup(guid, "data transfer");
        auto task = apex::new_task("data xfer", 0, parent);
        const auto getspace = [](tasktimer_execution_space_p& space) {
            std::stringstream ss;
            ss << (space->type == TASKTIMER_DEVICE_CPU ? "CPU " : "GPU ");
            ss << space->device_id << "," << space->instance_id;
            return ss.str();
        };
        /* source_type */
        task->arguments.push_back(apex::task_wrapper::argument(std::string(getspace(source_type))));
        task->argument_types.push_back(APEX_STRING_TYPE);
        task->argument_names.push_back("source_type");
        /* source_name */
        task->arguments.push_back(apex::task_wrapper::argument(std::string(source_name)));
        task->argument_types.push_back(APEX_STRING_TYPE);
        task->argument_names.push_back("source_name");
        /* source_ptr */
        task->arguments.push_back(apex::task_wrapper::argument((void*)source_ptr));
        task->argument_types.push_back(APEX_POINTER_TYPE);
        task->argument_names.push_back("source_ptr");
        /* dest_type */
        task->arguments.push_back(apex::task_wrapper::argument(std::string(getspace(dest_type))));
        task->argument_types.push_back(APEX_STRING_TYPE);
        task->argument_names.push_back("dest_type");
        /* dest_name */
        task->arguments.push_back(apex::task_wrapper::argument(std::string(dest_name)));
        task->argument_types.push_back(APEX_STRING_TYPE);
        task->argument_names.push_back("dest_name");
        /* dest_ptr */
        task->arguments.push_back(apex::task_wrapper::argument((void*)dest_ptr));
        task->argument_types.push_back(APEX_POINTER_TYPE);
        task->argument_names.push_back("dest_ptr");
        timerStack(task, true);
    }

    void tasktimer_data_transfer_stop_impl(tasktimer_guid_t guid) {
        APEX_UNUSED(guid);
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

