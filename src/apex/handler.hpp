/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <string>
#include <iostream>
#include <chrono>
#include <atomic>
#if defined(_MSC_VER) || defined(__APPLE__)
#include <thread>
#include <condition_variable>
#else
#include "pthread_wrapper.hpp"
#endif
#include "utils.hpp"
#include "apex_options.hpp"
#include "thread_instance.hpp"

namespace apex {

class handler
{
private:
#if defined(_MSC_VER) || defined(__APPLE__)
    std::condition_variable cv;
    std::mutex cv_m;
    static std::chrono::microseconds default_period;
    void _threadfunc(void) {
        while (!_terminate) {
            std::unique_lock<std::mutex> lk(cv_m);
            auto now = std::chrono::system_clock::now();
            auto rc = cv.wait_until(lk, now + _period);
            if(rc == std::cv_status::timeout) {
                //std::cerr << "Thread timed out.\n";
                _handler();
            } else {
                //std::cerr << "Thread cancelled.\n";
                return;
            }
        }
    };
#else
    static const unsigned int default_period = 100000;
    static void* _threadfunc(void * _ptw) {
        pthread_wrapper* ptw = (pthread_wrapper*)_ptw;
        // make sure APEX knows this is NOT a worker thread.
        thread_instance::instance(false).set_worker(false);
        ptw->_running = true;
        if (apex_options::pin_apex_threads()) {
             set_thread_affinity();
        }
        while (ptw->wait()) {
            handler* context = (handler*)(ptw->get_context());
            context->_handler();
        }
        ptw->_running = false;
        return nullptr;
    };
#endif
protected:
#if defined(_MSC_VER) || defined(__APPLE__)
  std::chrono::microseconds _period;
  std::thread* _timer_thread;
#else
  unsigned int _period;
  pthread_wrapper* _timer_thread;
#endif
  std::atomic<bool> _handler_initialized;
  std::atomic<bool> _terminate;
  void run(void) {
#if defined(_MSC_VER) || defined(__APPLE__)
    _timer_thread = new std::thread(&handler::_threadfunc, this);
#else
    _timer_thread = new pthread_wrapper(&handler::_threadfunc, (void*)(this), _period);
#endif
  };
  void set_timeout(unsigned int timeout) {
#if !defined(_MSC_VER) && !defined(__APPLE__)
    _period = timeout;
    _timer_thread->set_timeout(_period);
#else
    _period = std::chrono::microseconds(timeout);
#endif
  }
public:
  handler() :
      _period(default_period),
      _timer_thread(nullptr),
      _handler_initialized(false),
      _terminate(false)
    { }
  handler(unsigned int period) :
      _period(period),
      _timer_thread(nullptr),
      _handler_initialized(false),
      _terminate(false)
    { }
  void cancel(void) {
      _terminate = true;
      if(_timer_thread != nullptr) {
#if defined(_MSC_VER) || defined(__APPLE__)
        cv.notify_all();
        if (_timer_thread->joinable()) {
            _timer_thread->join();
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
      return true;
  };
};

}

