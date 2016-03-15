//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef APEX_HANDLER_H
#define APEX_HANDLER_H

#include <string>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace apex {

class handler
{
private:
    std::condition_variable cv;
    std::mutex cv_m;
    static std::chrono::microseconds default_period;
    void _threadfunc(void) {
        while (true) {
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
protected:
  std::chrono::microseconds _period;
  bool _handler_initialized;
  bool _terminate;
  std::thread* _timer_thread;
  void run(void) {
    _timer_thread = new std::thread(&handler::_threadfunc, this);
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
        cv.notify_all();
        _timer_thread->join();
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

#endif // APEX_HANDLER_H
