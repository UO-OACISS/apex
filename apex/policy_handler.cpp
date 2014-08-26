//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "policy_handler.hpp"
#include "thread_instance.hpp"

#include <iostream>

using namespace std;

namespace apex {

policy_handler::policy_handler (void) : handler() {
  _init();
}

policy_handler::policy_handler (unsigned int period) : handler(period) {
  _init();
}

void policy_handler::_handler(void) {
  periodic_event_data data;
  this->on_event(&data);
  this->reset();
}

void policy_handler::_init(void) {
  next_id = 0;
  _timer.async_wait(boost::bind(&policy_handler::_handler, this));
  run();
  return;
}

inline void policy_handler::reset(void) {
  if (!_terminate) {
    _timer.expires_at(_timer.expires_at() + boost::posix_time::microseconds(_period));
    _timer.async_wait(boost::bind(&policy_handler::_handler, this));
  }
}

int policy_handler::register_policy(const std::set<_event_type> & when,
    bool (*test_function)(void* arg1),
    void (*action_function)(void* arg2)) {
  int id = next_id++;
  policy_instance * instance = new policy_instance(id, test_function,
    action_function);
  for(event_type type : when) {
    switch(type) {
      case STARTUP: {
        startup_mutex.lock();
        startup_policies.push_back(instance);
        startup_mutex.unlock();
        break;
      }
      case SHUTDOWN: {
        shutdown_mutex.lock();
        shutdown_policies.push_back(instance);
        shutdown_mutex.unlock();
        break;
      }
      case NEW_NODE: {
        new_node_mutex.lock();
        new_node_policies.push_back(instance);
        new_node_mutex.unlock();
        break;
      }
      case NEW_THREAD: {
        new_thread_mutex.lock();
        new_thread_policies.push_back(instance);
        new_thread_mutex.unlock();
        break;
      }
      case START_EVENT: {
        start_event_mutex.lock();
        start_event_policies.push_back(instance);
        start_event_mutex.unlock();
        break;
      }
      case STOP_EVENT: {
        stop_event_mutex.lock();
        stop_event_policies.push_back(instance);
        stop_event_mutex.unlock();
        break;
      }
      case SAMPLE_VALUE: {
        sample_value_mutex.lock();
        sample_value_policies.push_back(instance);
        sample_value_mutex.unlock();
        break;
      }
      case PERIODIC: {
        periodic_mutex.lock();
        periodic_policies.push_back(instance);
        periodic_mutex.unlock();
        break;
      }

    }
  }
  return id;

}

void policy_handler::call_policies(const std::list<policy_instance*> & policies,
                   event_data* event_data_) {
  for(const policy_instance * policy : policies) {
    const bool result = policy->test_function(event_data_);
    if(result) {
      policy->action_function(event_data_);
    }
  }
}

void policy_handler::on_event(event_data* event_data_) {
  unsigned int tid = thread_instance::get_id();
  if (!_terminate) {
    switch(event_data_->event_type_) {
    case STARTUP: {
        startup_mutex.lock_shared();
        const std::list<policy_instance*> policies(startup_policies);
        startup_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case SHUTDOWN: {
    	_terminate = true;
        shutdown_mutex.lock_shared();
        const std::list<policy_instance*> policies(shutdown_policies);
        shutdown_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case NEW_NODE: {
        new_node_mutex.lock_shared();
        const std::list<policy_instance*> policies(new_node_policies);
        new_node_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case NEW_THREAD: {
        new_thread_mutex.lock_shared();
        const std::list<policy_instance*> policies(new_thread_policies);
        new_thread_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case START_EVENT: {
        start_event_mutex.lock_shared();
        const std::list<policy_instance*> policies(start_event_policies);
        start_event_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case STOP_EVENT: {
        stop_event_mutex.lock_shared();
        const std::list<policy_instance*> policies(stop_event_policies);
        stop_event_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case SAMPLE_VALUE: {
        sample_value_mutex.lock_shared();
        const std::list<policy_instance*> policies(sample_value_policies);
        sample_value_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    case PERIODIC: {
        periodic_mutex.lock_shared();
        const std::list<policy_instance*> policies(periodic_policies);
        periodic_mutex.unlock_shared();
        call_policies(policies, event_data_);
    	break;
    }
    } //end switch
  } // end if
  return;
}

} // end namespace apex

