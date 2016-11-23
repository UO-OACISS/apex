//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <string>
#include <iostream>
#include <chrono>
#include <atomic>
#include "pthread_wrapper.hpp"
#include "utils.hpp"
#include "apex_options.hpp"

namespace apex {

class handler
{
private:
    static const unsigned int default_period = 100000;
    static void* _threadfunc(void * _ptw) {
        pthread_wrapper* ptw = (pthread_wrapper*)_ptw;
        if (apex_options::pin_apex_threads()) {
             set_thread_affinity();
        }
        while (ptw->wait()) {
            handler* context = (handler*)(ptw->get_context());
            context->_handler();
        }
        return nullptr;
    };
protected:
  unsigned int _period;
  std::atomic<bool> _handler_initialized;
  std::atomic<bool> _terminate;
  pthread_wrapper* _timer_thread;
  void run(void) {
    _timer_thread = new pthread_wrapper(&handler::_threadfunc, (void*)(this), _period);
  };
public:
  handler() : 
      _period(default_period), 
      _handler_initialized(false), 
      _terminate(false), 
      _timer_thread(nullptr)
    { }
  handler(unsigned int period) : 
      _period(period), 
      _handler_initialized(false), 
      _terminate(false), 
      _timer_thread(nullptr)
    { }
  void cancel(void) {
      _terminate = true; 
      if(_timer_thread != nullptr) {
        _timer_thread->stop_thread();
        delete(_timer_thread);
        _timer_thread = nullptr;
      }
  }
  // virtual destructor
  virtual ~handler() {
      cancel();
  };
  // all methods in the interface that a handler has to override
  virtual bool _handler(void) {
      std::cout << "Default handler" << std::endl;
      return true;
  };
};

}

