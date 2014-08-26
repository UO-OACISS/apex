//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TAUHANDLER_HPP
#define TAUHANDLER_HPP

#include "event_listener.hpp"

using namespace std;

namespace apex {

class tau_listener : public event_listener {
private:
  void _init(void);
  bool _terminate;
public:
  tau_listener (void);
  ~tau_listener (void) { };
  void on_event(event_data* event_data_);
};

}

#endif // TAUHANDLER_HPP
