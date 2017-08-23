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

  std::unordered_map<std::string, task_identifier*>& task_identifier::get_task_id_name_map(void) {
      static APEX_NATIVE_TLS std::unordered_map<std::string, task_identifier*> task_id_name_map;
      return task_id_name_map;
  }
  std::unordered_map<uint64_t, task_identifier*>& task_identifier::get_task_id_addr_map(void) {
      static APEX_NATIVE_TLS std::unordered_map<uint64_t, task_identifier*> task_id_addr_map;
      return task_id_addr_map;
  }

  task_identifier * task_identifier::get_task_id (apex_function_address a, void** data_ptr) {
      if (data_ptr == 0) {
          auto& task_id_addr_map = get_task_id_addr_map();
          std::unordered_map<uint64_t,task_identifier*>::const_iterator got = task_id_addr_map.find (a);
          if ( got != task_id_addr_map.end() ) {
              return got->second;
          } else {
              task_identifier * tmp = new task_identifier(a, data_ptr);
              tmp->permanent = true;
              task_id_addr_map[a] = tmp;
              return tmp;
          }
      }
      return new task_identifier(a, data_ptr);
  }

  task_identifier * task_identifier::get_task_id (const std::string& n, void** data_ptr) {
      if (data_ptr == 0) {
          auto& task_id_name_map = get_task_id_name_map();
          std::unordered_map<std::string,task_identifier*>::const_iterator got = task_id_name_map.find (n);
          if ( got != task_id_name_map.end() ) {
              return got->second;
          } else {
              task_identifier * tmp = new task_identifier(n, data_ptr);
              tmp->permanent = true;
              task_id_name_map.insert(std::pair<std::string,task_identifier*>(n, tmp));
              return tmp;
          }
      }
      return new task_identifier(n, data_ptr);
  }
}

