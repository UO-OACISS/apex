//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef APEX_HAVE_HPX3
#include <hpx/hpx.hpp>
#include <hpx/util/interval_timer.hpp>
#endif

#include "apex_types.h"
#include "handler.hpp"
#include "event_listener.hpp"
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <functional>
#include <chrono>
#include <memory>
#include <array>

#if __cplusplus > 201701L 
#include <shared_mutex>
#elif __cplusplus > 201402L
#include <shared_lock>
#else
#include <mutex>
#endif

#ifdef SIGEV_THREAD_ID
#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif /* ifndef sigev_notify_thread_id */
#endif /* ifdef SIGEV_THREAD_ID */

namespace apex
{

class policy_instance
{
public:
    int id;
    std::function<bool(apex_context const&)> func;
    policy_instance(int id_, std::function<bool(apex_context const&)> func_) : id(id_),
        func(func_) {};
};

class policy_handler : public handler, public event_listener
{
private:
#if __cplusplus > 201701L 
    typedef std::shared_mutex mutex_type;
#elif __cplusplus > 201402L
    typedef std::shared_lock mutex_type;
#else
    typedef std::mutex mutex_type;
#endif
    using policy_list = std::list<std::shared_ptr<policy_instance> >;

    void _init(void);
    policy_list startup_policies;
    policy_list shutdown_policies;
    policy_list new_node_policies;
    policy_list new_thread_policies;
    policy_list exit_thread_policies;
    policy_list start_event_policies;
    policy_list stop_event_policies;
    policy_list yield_event_policies;
    policy_list resume_event_policies;
    policy_list new_task_event_policies;
    policy_list destroy_task_event_policies;
    policy_list new_dependency_event_policies;
    policy_list satisfy_dependency_event_policies;
    policy_list set_task_state_event_policies;
    policy_list acquire_data_event_policies;
    policy_list release_data_event_policies;
    policy_list new_event_event_policies;
    policy_list destroy_event_event_policies;
    policy_list new_data_event_policies;
    policy_list destroy_data_event_policies;
    policy_list sample_value_policies;
    policy_list periodic_policies;
    std::array<policy_list,APEX_MAX_EVENTS > custom_event_policies;
    mutex_type startup_mutex;
    mutex_type shutdown_mutex;
    mutex_type new_node_mutex;
    mutex_type new_thread_mutex;
    mutex_type exit_thread_mutex;
    mutex_type start_event_mutex;
    mutex_type stop_event_mutex;
    mutex_type yield_event_mutex;
    mutex_type resume_event_mutex;
    mutex_type new_task_event_mutex;
    mutex_type destroy_task_event_mutex;
    mutex_type new_dependency_event_mutex;
    mutex_type satisfy_dependency_event_mutex;
    mutex_type set_task_state_event_mutex;
    mutex_type acquire_data_event_mutex;
    mutex_type release_data_event_mutex;
    mutex_type new_event_event_mutex;
    mutex_type destroy_event_event_mutex;
    mutex_type new_data_event_mutex;
    mutex_type destroy_data_event_mutex;
    mutex_type sample_value_mutex;
    mutex_type custom_event_mutex;
    mutex_type periodic_mutex;
    void call_policies(
        const std::list<std::shared_ptr<policy_instance> > & policies,
        event_data &event_data);
#ifdef APEX_HAVE_HPX3
    hpx::util::interval_timer hpx_timer;
#endif
public:
    policy_handler (void);
/*
    template<Rep, Period>
    policy_handler (std::chrono::duration<Rep, Period> const& period);
*/
    policy_handler(uint64_t period_microseconds);
    ~policy_handler (void) { };
    void on_startup(startup_event_data &data);
    void on_shutdown(shutdown_event_data &data);
    void on_new_node(node_event_data &data);
    void on_new_thread(new_thread_event_data &data);
    void on_exit_thread(event_data &data);
    bool on_start(task_identifier *id);
    void on_stop(std::shared_ptr<profiler> &p);
    void on_yield(std::shared_ptr<profiler> &p);
    bool on_resume(task_identifier * id);
    void on_new_task(new_task_event_data & data);
    void on_destroy_task(destroy_task_event_data & data);
    void on_new_dependency(new_dependency_event_data & data);
    void on_satisfy_dependency(satisfy_dependency_event_data & data);
    void on_set_task_state(set_task_state_event_data &data);
    void on_acquire_data(acquire_data_event_data &data);
    void on_release_data(release_data_event_data &data);
    void on_new_event(new_event_event_data &data);
    void on_destroy_event(destroy_event_event_data &data);
    void on_new_data(new_data_event_data &data);
    void on_destroy_data(destroy_data_event_data &data);
    void on_sample_value(sample_value_event_data &data);
    void on_custom_event(custom_event_data &data);
    void on_periodic(periodic_event_data &data);

    int register_policy(const apex_event_type & when,
                        std::function<int(apex_context const&)> f);
    int deregister_policy(apex_policy_handle * handle);
    bool _handler(void);
    void _reset(void);
};

}

