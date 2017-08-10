//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "task_identifier.hpp"
#include "thread_instance.hpp"

#if APEX_HAVE_BFD
#include "address_resolution.hpp"
#ifdef __MIC__
#include <boost/regex.hpp>
#define REGEX_NAMESPACE boost
#else
#include <regex>
#define REGEX_NAMESPACE std
#endif
#endif

namespace apex {

const std::string& task_identifier::get_name(bool resolve) {
    if (!has_name && resolve) {
      if (_resolved_name == "" && address != APEX_NULL_FUNCTION_ADDRESS) {
        //_resolved_name = lookup_address((uintptr_t)address, false);         
        _resolved_name = thread_instance::instance().map_addr_to_name(address);
      }
      _resolved_name.assign(demangle(_resolved_name));
      return _resolved_name;
    } else {
        if (resolve) {
#ifdef APEX_HAVE_BFD
            REGEX_NAMESPACE::regex rx (".*UNRESOLVED ADDR (.*)");
            if (REGEX_NAMESPACE::regex_match (name,rx)) {
                const REGEX_NAMESPACE::regex separator(" ADDR ");
                REGEX_NAMESPACE::sregex_token_iterator token(name.begin(), name.end(), separator, -1);
                *token++; // ignore
                std::string addr_str = *token++;
                void* addr_addr;
                sscanf(addr_str.c_str(), "%p", &addr_addr);
                std::string * tmp = lookup_address((uintptr_t)addr_addr, true);
                REGEX_NAMESPACE::regex old_address("UNRESOLVED ADDR " + addr_str);
                name = REGEX_NAMESPACE::regex_replace(name, old_address, demangle(*tmp));
            }
#endif
            name.assign(demangle(name));
        }
      return name;
    }
  }

}

