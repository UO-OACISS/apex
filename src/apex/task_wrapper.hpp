/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

namespace apex {
struct task_wrapper;
}

#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#include <hpx/util/external_timer.hpp>
#endif
#include "task_identifier.hpp"
#include "profiler.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>

namespace apex {

/**
  \brief A wrapper around APEX tasks.
  */
#ifdef APEX_HAVE_HPX_CONFIG
struct task_wrapper : public hpx::util::external_timer::task_wrapper {
#else
struct task_wrapper {
#endif
/**
  \brief A pointer to the task_identifier for this task_wrapper.
  */
    task_identifier * task_id;
/**
  \brief A pointer to the active profiler object timing this task.
  */
    profiler * prof;
/**
  \brief An internally generated GUID for this task.
  */
    uint64_t guid;
/**
  \brief An internally generated GUID for the parent task of this task.
  */
    uint64_t parent_guid;
/**
  \brief A managed pointer to the parent task_wrapper for this task.
  */
    std::shared_ptr<task_wrapper> parent;
/**
  \brief Internal usage, used to manage HPX direct actions when their
         parent task is yielded by the runtime.
  */
    std::vector<profiler*> data_ptr;
/**
  \brief If the task changes names
         after creation (due to the application of an annotation) then the alias
         becomes the new task_identifier for the task.
  */
    task_identifier* alias;
/**
  \brief Constructor.
  */
    task_wrapper(void) :
        task_id(nullptr),
        prof(nullptr),
        guid(0ull),
        parent_guid(0ull),
        parent(nullptr),
        alias(nullptr)
    { }
/**
  \brief Get the task_identifier for this task_wrapper.
  \returns A pointer to the task_identifier
  */
    inline task_identifier * get_task_id(void) {
        if (alias != nullptr) {
            return alias;
        }
        return task_id;
    }
/**
  \brief Static method to get a pre-defined task_wrapper around "main".
  \returns A shared pointer to the task_wrapper
  */
    static std::shared_ptr<task_wrapper> & get_apex_main_wrapper(void) {
        static std::shared_ptr<task_wrapper> tt_ptr(nullptr);
        static std::mutex mtx;
        if (tt_ptr.get() == nullptr) {
            mtx.lock();
            if (tt_ptr.get() == nullptr) {
                const std::string apex_main_str("APEX MAIN");
                tt_ptr = std::make_shared<task_wrapper>();
                tt_ptr->task_id = task_identifier::get_task_id(apex_main_str);
            }
            mtx.unlock();
        }
        return tt_ptr;
    }
}; // struct task_wrapper

} // namespace apex
