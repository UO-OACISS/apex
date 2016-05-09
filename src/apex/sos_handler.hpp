//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "handler.hpp"
#include <memory>
#include <mutex>
#include "sos.h"

namespace apex {

class sos_handler : public handler {
private:
  bool _terminate;
  SOS_runtime * _runtime;
  SOS_pub * _pub;
  void _make_pub (void);
  char* timer;
  char* counter;
  std::mutex _terminate_mutex;
public:
  sos_handler (int argc, char * argv[], int period);
  ~sos_handler (void);
  virtual bool _handler(void);
  bool _handler_internal(void);
  void terminate (void);
};

}

