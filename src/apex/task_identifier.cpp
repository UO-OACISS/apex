//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "task_identifier.hpp"
#include "thread_instance.hpp"

namespace apex {

std::string task_identifier::get_name() {
    if (!has_name) {
      if (_resolved_name == "" && address != APEX_NULL_FUNCTION_ADDRESS) {
        //_resolved_name = lookup_address((uintptr_t)address, false);         
        _resolved_name = thread_instance::instance().map_addr_to_name(address);
      }
      return demangle(_resolved_name);
    }
    return name;
  }

}

