//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <string>
#include <iostream>
#include <chrono>
#ifdef APEX_HAVE_HPX3 
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#else
#include "pthread_wrapper.hpp"
#endif
#include "utils.hpp"
#include "apex_options.hpp"

namespace apex {

class handler
{
private:
    static const unsigned int default_period = 100000;
#ifdef APEX_HAVE_HPX3 
    static boost::asio::io_service _io;
    void _threadfunc(void) {
        _io.run();
    }
#else
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
#endif
protected:
  unsigned int _period;
  bool _handler_initialized;
  bool _terminate;
#ifdef APEX_HAVE_HPX3 
  boost::asio::deadline_timer _timer;
  boost::thread* _timer_thread;
  void reset(void) {
      if (!_terminate) {
          _timer.expires_at(_timer.expires_at() + boost::posix_time::microseconds(_period));
          _timer.async_wait(boost::bind(&handler::_handler, this));
      }
  }
#else
  pthread_wrapper* _timer_thread;
#endif
  void run(void) {
#ifdef APEX_HAVE_HPX3 
    _timer_thread = new boost::thread(&handler::_threadfunc, this);
#else
    _timer_thread = new pthread_wrapper(&handler::_threadfunc, (void*)(this), _period);
#endif
  };
public:
  handler() : 
      _period(default_period), 
      _handler_initialized(false), 
      _terminate(false), 
#ifdef APEX_HAVE_HPX3 
      _timer(_io, boost::posix_time::microseconds(_period)),
#endif
      _timer_thread(nullptr)
    { }
  handler(unsigned int period) : 
      _period(period), 
      _handler_initialized(false), 
      _terminate(false), 
#ifdef APEX_HAVE_HPX3 
      _timer(_io, boost::posix_time::microseconds(_period)),
#endif
      _timer_thread(nullptr)
    { }
  void cancel(void) {
      _terminate = true; 
      if(_timer_thread != nullptr) {
#ifdef APEX_HAVE_HPX3 
        _timer.cancel();
        if (_timer_thread->try_join_for(boost::chrono::seconds(1))) {
            _timer_thread->interrupt();
        }
#else
        _timer_thread->stop_thread();
#endif
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
#ifdef APEX_HAVE_HPX3 
      this->reset();
#endif
      return true;
  };
};

}

