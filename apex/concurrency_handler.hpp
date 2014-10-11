//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CONCURRENCYHANDLER_HPP
#define CONCURRENCYHANDLER_HPP

#include "handler.hpp"
#include "event_listener.hpp"
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <boost/thread/mutex.hpp>

#ifdef SIGEV_THREAD_ID
#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif /* ifndef sigev_notify_thread_id */
#endif /* ifdef SIGEV_THREAD_ID */

//using namespace std;

namespace apex {

class concurrency_handler : public handler, public event_listener {
private:
  void _init(void);
  // vectors and mutex
  std::vector<std::stack<std::string>* > _event_stack;
  boost::mutex _vector_mutex;
  // periodic samples of stack top states
  std::vector<std::map<std::string, unsigned int>* > _states;
  // functions and mutex
  std::set<std::string> _functions;
  boost::mutex _function_mutex;
  int _option;
public:
  concurrency_handler (void);
  concurrency_handler (char *option);
  concurrency_handler (unsigned int period);
  concurrency_handler (unsigned int period, char* option);
  ~concurrency_handler (void) { };
  void on_startup(startup_event_data &event_data) { };
  void on_shutdown(shutdown_event_data &event_data);
  void on_new_node(node_event_data &event_data) { };
  void on_new_thread(new_thread_event_data &event_data);
  void on_start(timer_event_data &event_data);
  void on_stop(timer_event_data &event_data);
  void on_resume(timer_event_data &event_data);
  void on_sample_value(sample_value_event_data &event_data) { };
  void on_periodic(periodic_event_data &event_data) { };

  void _handler(void);
  std::stack<std::string>* get_event_stack(unsigned int tid);
  void add_thread(unsigned int tid) ;
  void reset(void);
  void output_samples(int node_id);
};

}

#endif // CONCURRENCYHANDLER_HPP
