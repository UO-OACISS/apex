//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <string>
#include <iostream>
#include <chrono>
#include "pthread_wrapper.hpp"

namespace apex {

class handler
{
private:
    static unsigned int default_period;
    static void* _threadfunc(void * _ptw) {
        pthread_wrapper* ptw = (pthread_wrapper*)_ptw;
        while (ptw->wait()) {
            handler* context = (handler*)(ptw->get_context());
            context->_handler();
        }
        return nullptr;
    };
protected:
  unsigned int _period;
  bool _handler_initialized;
  bool _terminate;
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

