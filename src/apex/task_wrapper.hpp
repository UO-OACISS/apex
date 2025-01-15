/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
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
#include <hpx/modules/threading_base.hpp>
#endif
#include "task_identifier.hpp"
#include "profiler.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include "dependency_tree.hpp"
#include "apex_clock.hpp"
#include <variant>

// forward declaration...
namespace apex {
APEX_EXPORT void internal_destroy(task_wrapper * tt_ptr);
}

namespace apex {

/**
  \brief A wrapper around APEX tasks.
  */
#ifdef APEX_HAVE_HPX_CONFIG
struct task_wrapper : public hpx::util::external_timer::task_wrapper,
#else
struct task_wrapper :
#endif
public std::enable_shared_from_this<task_wrapper>
{
    typedef enum e_task_state {
        CREATED,
        RUNNING,
        YIELDED,
        STOPPED,
        DESTROYED
    } task_state_t;
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
  \brief A managed pointer to the parent task_wrapper for this task.
  */
    std::vector<std::shared_ptr<task_wrapper>> parents;
/**
  \brief A node in the task tree representing this task type
  */
    std::shared_ptr<dependency::Node> tree_node;
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
  \brief Thread ID of the thread that created this task.
  */
    long unsigned int thread_id;
/**
  \brief Time (in microseconds) when this task was created
  */
    uint64_t create_ns;
/**
  \brief Time (in microseconds) when this task was _first_ started
  */
  private:
    std::vector<uint64_t> start_ns;
    public:
/**
  \brief Whether this event requires separate start/end events in gtrace
  */
    bool explicit_trace_start;
    task_state_t state;
    bool implicit_parent;
    std::shared_ptr<task_wrapper> previous_task;
/**
    \brief Vector of arguments to the task, optional data from taskStubs implementation
           only used by the google trace events format output.
  */
    using argument = std::variant<uint64_t,int64_t,double,std::string,void*>;
    std::vector<apex_argument_type_t> argument_types;
    std::vector<argument> arguments;
    std::vector<std::string> argument_names;
    uint64_t trace_tid;
    uint64_t is_async;
    bool destroying;
/**
  \brief Constructor.
  */
    task_wrapper(void) :
        task_id(nullptr),
        prof(nullptr),
        guid(0ull),
        parents({}),
        tree_node(nullptr),
        alias(nullptr),
        thread_id(0UL),
        create_ns(our_clock::now_ns()),
        explicit_trace_start(false),
        state(CREATED),
        implicit_parent(true),
        previous_task(nullptr),
        is_async(false),
        destroying(false)
    { }
/**
  \brief Destructor.
  */
    ~task_wrapper(void) {
        // don't delete the alias, because task_identifier objects never go away
        /*
        if (this->state == STOPPED && !this->destroying) {
            destroying = true;
            internal_destroy(this);
        }
        */
    }

    std::shared_ptr<task_wrapper> not_dangerous()
    {
        return shared_from_this();
    }

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
                const std::string apex_main_str(APEX_MAIN_STR);
                tt_ptr = std::make_shared<task_wrapper>();
                tt_ptr->task_id = task_identifier::get_task_id(apex_main_str);
                std::shared_ptr<dependency::Node> tmp(new dependency::Node(tt_ptr->task_id, nullptr));
                tt_ptr->tree_node = tmp;
            }
            mtx.unlock();
        }
        return tt_ptr;
    }
    void assign_heritage() {
        // make/find a node for ourselves
        //tree_node = parents[0]->tree_node->appendChild(task_id);
        for(auto& parent : parents) {
            tree_node = parent->tree_node->appendChild(task_id, tree_node);
        }
    }
    void update_heritage() {
        // make/find a node for ourselves
        //tree_node = parents[0]->tree_node->replaceChild(task_id, alias);
        for(auto& parent : parents) {
            tree_node = parent->tree_node->replaceChild(task_id, alias);
        }
    }
    bool is_parent(std::shared_ptr<task_wrapper> candidate) {
        for (auto& p : parents) {
            /*printf("is '%s' parent of '%s'?...\n",
            candidate->task_id->get_name().c_str(),
            this->task_id->get_name().c_str());*/
            if (p == candidate) return true;
        }
        return false;
    }
    bool is_ancestor(std::shared_ptr<task_wrapper> candidate) {
        // am I the parent of the candidate?
        if (is_parent(candidate)) return true;
        // are one of my parents an ancestor of the candidate?
        for (auto& p : parents) {
            if (p->is_ancestor(candidate)) return true;
        }
        return false;
    }
    double get_create_us() {
        return double(create_ns) * 1.0e-3;
    }
    uint64_t get_create_ns() {
        return create_ns;
    }
    double get_start_us() {
        return double(get_start_ns()) * 1.0e-3;
    }
    uint64_t get_start_ns() {
        return start_ns.back();
    }
    double get_flow_us(uint64_t ts) {
        return double(get_flow_ns(ts)) * 1.0e-3;
    }
    double get_flow_ns(uint64_t ts) {
        uint64_t prior = start_ns.front();
        for (auto start : start_ns) {
            if (ts < start) {
                break;
            }
            prior = start+1;
        }
        return (double)prior;
    }
    void assign_start_ns(double start) {
        start_ns.push_back(start);
    }
}; // struct task_wrapper

} // namespace apex
