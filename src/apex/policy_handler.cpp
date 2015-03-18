//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#include <hpx/include/runtime.hpp>
#endif

#include "apex.hpp"
#include "policy_handler.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <boost/make_shared.hpp>

using namespace std;

namespace apex {

#ifdef APEX_HAVE_HPX3
policy_handler::policy_handler (void) : handler() { }
#else
policy_handler::policy_handler (void) : handler() { }
#endif

/*
template <typename Rep, typename Period>
policy_handler::policy_handler (duration<Rep, Period> const& period) : handler(period)
{
    _init();
}
*/

#ifdef APEX_HAVE_HPX3
policy_handler::policy_handler (uint64_t period_microseconds) : handler(period_microseconds), hpx_timer(boost::bind(&policy_handler::_handler, this), _period, "apex_policy_handler") 
{
    _init();
}
#else
policy_handler::policy_handler (uint64_t period_microseconds) : handler(period_microseconds) 
{
    _init();
}
#endif

bool policy_handler::_handler(void) {
  static int dummy = initialize_worker_thread_for_TAU();
  APEX_UNUSED(dummy);
  if (_terminate) return true;
  periodic_event_data data;
  this->on_periodic(data);
  this->_reset();
  return true;
}

void policy_handler::_init(void) {
  next_id = 0;
#ifdef APEX_HAVE_HPX3
  hpx_timer.start();
#else
  _timer.async_wait(boost::bind(&policy_handler::_handler, this));
  run();
#endif
  return;
}

inline void policy_handler::_reset(void) {
#ifdef APEX_HAVE_HPX3
  if (_terminate) {
    hpx_timer.stop();
  }
#else
  if (!_terminate) {
    _timer.expires_at(_timer.expires_at() + boost::posix_time::microseconds(_period));
    _timer.async_wait(boost::bind(&policy_handler::_handler, this));
  }
#endif
}

int policy_handler::register_policy(const apex_event_type & when,
    std::function<int(apex_context const&)> f) {
  int id = next_id++;
  boost::shared_ptr<policy_instance> instance(
    boost::make_shared<policy_instance>(id, f));
    switch(when) {
      case APEX_STARTUP: {
        boost::unique_lock<mutex_type> l(startup_mutex);
        startup_policies.push_back(instance);
        break;
      }
      case APEX_SHUTDOWN: {
        boost::unique_lock<mutex_type> l(shutdown_mutex);
        shutdown_policies.push_back(instance);
        break;
      }
      case APEX_NEW_NODE: {
        boost::unique_lock<mutex_type> l(new_node_mutex);
        new_node_policies.push_back(instance);
        break;
      }
      case APEX_NEW_THREAD: {
        boost::unique_lock<mutex_type> l(new_thread_mutex);
        new_thread_policies.push_back(instance);
        break;
      }
      case APEX_START_EVENT: {
        boost::unique_lock<mutex_type> l(start_event_mutex);
        start_event_policies.push_back(instance);
        break;
      }
      case APEX_RESUME_EVENT: {
        boost::unique_lock<mutex_type> l(resume_event_mutex);
        resume_event_policies.push_back(instance);
        break;
      }
      case APEX_STOP_EVENT: {
        boost::unique_lock<mutex_type> l(stop_event_mutex);
        stop_event_policies.push_back(instance);
        break;
      }
      case APEX_SAMPLE_VALUE: {
        boost::unique_lock<mutex_type> l(sample_value_mutex);
        sample_value_policies.push_back(instance);
        break;
      }
      case APEX_CUSTOM_EVENT: {
        boost::unique_lock<mutex_type> l(custom_event_mutex);
        custom_event_policies.push_back(instance);
        break;
      }
      case APEX_PERIODIC: {
        boost::unique_lock<mutex_type> l(periodic_mutex);
        periodic_policies.push_back(instance);
        break;
      }
  }
  return id;

}

inline void policy_handler::call_policies(
    const std::list<boost::shared_ptr<policy_instance> > & policies,
    event_data &event_data) {
  for(const boost::shared_ptr<policy_instance>& policy : policies) {
    apex_context my_context;
    my_context.event_type = event_data.event_type_;
    my_context.policy_handle = NULL;
    if (event_data.event_type_ == APEX_CUSTOM_EVENT) {
        my_context.data = event_data.data;
    } else {
        my_context.data = NULL;
    }
    const bool result = policy->func(my_context);
    if(result != APEX_NOERROR) {
      printf("Warning: registered policy function failed!\n");
    }
  }
}

void policy_handler::on_startup(startup_event_data &event_data) {
  if (_terminate) return;
            if (startup_policies.empty())
                return;
        call_policies(startup_policies, event_data);
}

void policy_handler::on_shutdown(shutdown_event_data &event_data) {
  if (_terminate) return;
        _terminate = true;
            if (shutdown_policies.empty())
                return;
        call_policies(shutdown_policies, event_data);
}

void policy_handler::on_new_node(node_event_data &event_data) {
  if (_terminate) return;
            if (new_node_policies.empty())
                return;
        call_policies(new_node_policies, event_data);
}

void policy_handler::on_new_thread(new_thread_event_data &event_data) {
  if (_terminate) return;
            if (new_thread_policies.empty())
                return;
        call_policies(new_thread_policies, event_data);
}

void policy_handler::on_start(apex_function_address function_address, string *timer_name) {
  if (_terminate) return;
  if (start_event_policies.empty()) return;
        //call_policies(start_event_policies, event_data);
  for(const boost::shared_ptr<policy_instance>& policy : start_event_policies) {
    apex_context my_context;
    my_context.event_type = APEX_START_EVENT;
    my_context.policy_handle = NULL;
    const bool result = policy->func(my_context);
    if(result != APEX_NOERROR) {
      printf("Warning: registered policy function failed!\n");
    }
  }
  APEX_UNUSED(function_address);
  APEX_UNUSED(timer_name);
}

void policy_handler::on_stop(profiler *p) {
  if (_terminate) return;
            if (stop_event_policies.empty())
                return;
        //call_policies(stop_event_policies, event_data);
  for(const boost::shared_ptr<policy_instance>& policy : stop_event_policies) {
    apex_context my_context;
    my_context.event_type = APEX_STOP_EVENT;
    my_context.policy_handle = NULL;
    const bool result = policy->func(my_context);
    if(result != APEX_NOERROR) {
      printf("Warning: registered policy function failed!\n");
    }
  }
  APEX_UNUSED(p);
}

void policy_handler::on_resume(profiler * p) {
  if (_terminate) return;
  APEX_UNUSED(p);
}

void policy_handler::on_sample_value(sample_value_event_data &event_data) {
  if (_terminate) return;
            if (sample_value_policies.empty())
                return;
        call_policies(sample_value_policies, event_data);
}

void policy_handler::on_custom_event(custom_event_data &event_data) {
  if (_terminate) return;
            if (custom_event_policies.empty())
                return;
        call_policies(custom_event_policies, event_data);
}

void policy_handler::on_periodic(periodic_event_data &event_data) {
  if (_terminate) return;
            if (periodic_policies.empty())
                return;
        call_policies(periodic_policies, event_data);
}

} // end namespace apex

