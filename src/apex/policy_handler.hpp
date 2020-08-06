//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef APEX_HAVE_HPX
#include <hpx/hpx.hpp>
#include <hpx/modules/runtime_local.hpp>
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
#include "apex_cxx_shared_lock.hpp"

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
    void _init(void);
    std::list<std::shared_ptr<policy_instance> > startup_policies;
    std::list<std::shared_ptr<policy_instance> > shutdown_policies;
    std::list<std::shared_ptr<policy_instance> > new_node_policies;
    std::list<std::shared_ptr<policy_instance> > new_thread_policies;
    std::list<std::shared_ptr<policy_instance> > exit_thread_policies;
    std::list<std::shared_ptr<policy_instance> > start_event_policies;
    std::list<std::shared_ptr<policy_instance> > stop_event_policies;
    std::list<std::shared_ptr<policy_instance> > yield_event_policies;
    std::list<std::shared_ptr<policy_instance> > resume_event_policies;
    std::list<std::shared_ptr<policy_instance> > sample_value_policies;
    std::list<std::shared_ptr<policy_instance> > send_policies;
    std::list<std::shared_ptr<policy_instance> > recv_policies;
    std::list<std::shared_ptr<policy_instance> > periodic_policies;
    std::array<std::list<std::shared_ptr<policy_instance> >,APEX_MAX_EVENTS >
        custom_event_policies;
    shared_mutex_type startup_mutex;
    shared_mutex_type shutdown_mutex;
    shared_mutex_type new_node_mutex;
    shared_mutex_type new_thread_mutex;
    shared_mutex_type exit_thread_mutex;
    shared_mutex_type start_event_mutex;
    shared_mutex_type stop_event_mutex;
    shared_mutex_type yield_event_mutex;
    shared_mutex_type resume_event_mutex;
    shared_mutex_type sample_value_mutex;
    shared_mutex_type send_mutex;
    shared_mutex_type recv_mutex;
    shared_mutex_type custom_event_mutex;
    shared_mutex_type periodic_mutex;
    void call_policies(
        const std::list<std::shared_ptr<policy_instance> > & policies,
        void *event_data, const apex_event_type& event_type);
#ifdef APEX_HAVE_HPX
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
    void on_dump(dump_event_data &data);
    void on_reset(task_identifier * id)
        { APEX_UNUSED(id); };
    void on_pre_shutdown(void) {};
    void on_shutdown(shutdown_event_data &data);
    void on_new_node(node_event_data &data);
    void on_new_thread(new_thread_event_data &data);
    void on_exit_thread(event_data &data);
    bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
    void on_stop(std::shared_ptr<profiler> &p);
    void on_yield(std::shared_ptr<profiler> &p);
    bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
    void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
        APEX_UNUSED(tt_ptr);
    };
    void on_sample_value(sample_value_event_data &data);
    void on_custom_event(custom_event_data &data);
    void on_periodic(periodic_event_data &data);
    void on_send(message_event_data &data);
    void on_recv(message_event_data &data);
    void set_node_id(int node_id, int node_count) { APEX_UNUSED(node_id);
        APEX_UNUSED(node_count); }

    int register_policy(const apex_event_type & when,
                        std::function<int(apex_context const&)> f);
    int deregister_policy(apex_policy_handle * handle);
    bool _handler(void);
    void _reset(void);
};

}

