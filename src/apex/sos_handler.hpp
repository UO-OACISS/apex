//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "handler.hpp"
#include <memory>
#include "sos.h"
#include <mutex>

namespace apex {

class sos_handler : public handler {
private:
  bool _terminate;
  SOS_runtime * _runtime;
  SOS_pub * _pub;
  void _make_pub (void);
  SOS_val timer;
  SOS_val counter;
  std::mutex _terminate_mutex;
public:
  sos_handler (int argc, char * argv[], int period);
  ~sos_handler (void);
  virtual bool _handler(void);
  void terminate (void);
};

}

