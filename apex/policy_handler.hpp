//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef POLICYHANDLER_HPP
#define POLICYHANDLER_HPP

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

#include <boost/atomic/atomic.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/shared_ptr.hpp>

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
    typedef boost::shared_mutex mutex_type;

    void _init(void);
    std::list<boost::shared_ptr<policy_instance> > startup_policies;
    std::list<boost::shared_ptr<policy_instance> > shutdown_policies;
    std::list<boost::shared_ptr<policy_instance> > new_node_policies;
    std::list<boost::shared_ptr<policy_instance> > new_thread_policies;
    std::list<boost::shared_ptr<policy_instance> > start_event_policies;
    std::list<boost::shared_ptr<policy_instance> > stop_event_policies;
    std::list<boost::shared_ptr<policy_instance> > sample_value_policies;
    std::list<boost::shared_ptr<policy_instance> > periodic_policies;
    mutex_type startup_mutex;
    mutex_type shutdown_mutex;
    mutex_type new_node_mutex;
    mutex_type new_thread_mutex;
    mutex_type start_event_mutex;
    mutex_type stop_event_mutex;
    mutex_type sample_value_mutex;
    mutex_type periodic_mutex;
    void call_policies(
        const std::list<boost::shared_ptr<policy_instance> > & policies,
        event_data * event_data_);
    boost::atomic_int next_id;
public:
    policy_handler (void);
/*
    template<Rep, Period>
    policy_handler (std::chrono::duration<Rep, Period> const& period);
*/
    policy_handler(uint64_t period_microseconds);
    ~policy_handler (void) { };
    void on_event(event_data* event_data_);
    int register_policy(const apex_event_type & when,
                        std::function<bool(apex_context const&)> f);
    void _handler(void);
    void reset(void);
};

}

#endif // POLICYHANDLER_HPP
