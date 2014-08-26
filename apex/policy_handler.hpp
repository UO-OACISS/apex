//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef POLICYHANDLER_HPP
#define POLICYHANDLER_HPP

#include "handler.hpp"
#include "event_listener.hpp"
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <boost/thread/shared_mutex.hpp>

#ifdef SIGEV_THREAD_ID
#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif /* ifndef sigev_notify_thread_id */
#endif /* ifdef SIGEV_THREAD_ID */

using namespace std;

namespace apex {

struct policy_instance {
  int id;
  bool (*test_function)(void* arg1);
  void (*action_function)(void* arg2);

  policy_instance(int id_, bool (*test_function_)(void* arg1),
    void (*action_function_)(void* arg2)) : id(id_),
    test_function(test_function_), action_function(action_function_) {};
};

class policy_handler : public handler, public event_listener {
private:
  void _init(void);
  std::list<policy_instance*> startup_policies;
  std::list<policy_instance*> shutdown_policies;
  std::list<policy_instance*> new_node_policies;
  std::list<policy_instance*> new_thread_policies;
  std::list<policy_instance*> start_event_policies;
  std::list<policy_instance*> stop_event_policies;
  std::list<policy_instance*> sample_value_policies;
  std::list<policy_instance*> periodic_policies;
  boost::shared_mutex startup_mutex;
  boost::shared_mutex shutdown_mutex;
  boost::shared_mutex new_node_mutex;
  boost::shared_mutex new_thread_mutex;
  boost::shared_mutex start_event_mutex;
  boost::shared_mutex stop_event_mutex;
  boost::shared_mutex sample_value_mutex;
  boost::shared_mutex periodic_mutex;
  void call_policies(const std::list<policy_instance*> & policies,
                     event_data * event_data_);
  std::atomic_int next_id;
public:
  policy_handler (void);
  policy_handler (unsigned int period);
  ~policy_handler (void) { };
  void on_event(event_data* event_data_);
  int register_policy(const std::set<_event_type> & when,
                      bool (*test_function)(void* arg1),
                      void (*action_function)(void* arg2));
  void _handler(void);
  void reset(void);
};

}

#endif // POLICYHANDLER_HPP
