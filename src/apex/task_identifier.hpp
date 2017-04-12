//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex_types.h"
#include "utils.hpp"
#include <functional>
#include <string>

namespace apex {

class task_identifier {
public:
  apex_function_address address;
  std::string name;
  std::string _resolved_name;
  bool has_name;
  void** _data_ptr; // not included in comparisons!
  task_identifier(void) :
      address(0L), name(""), _resolved_name(""), has_name(false), _data_ptr(0) {};
  task_identifier(apex_function_address a, void** data_ptr = 0) :
      address(a), name(""), _resolved_name(""), has_name(false), _data_ptr(data_ptr) {};
  task_identifier(std::string n, void** data_ptr = 0) :
      address(0L), name(demangle(n)), _resolved_name(""), has_name(true), _data_ptr(data_ptr) {};
  task_identifier(const task_identifier& rhs) :
      address(rhs.address), name(rhs.name), _resolved_name(rhs._resolved_name), has_name(rhs.has_name), _data_ptr(rhs._data_ptr) { };
	  /*
  task_identifier(profiler * p) :
      address(0L), name(""), _resolved_name("") {
      if (p->have_name) {
          name = *p->timer_name;
          has_name = true;
      } else {
          address = p->action_address;
          has_name = false;
      }
  }
  */
  std::string get_name(bool resolve = true);
  ~task_identifier() { }
  // requried for using this class as a key in an unordered map.
  // the hash function is defined below.
  bool operator==(const task_identifier &other) const {
    return (address == other.address && name.compare(other.name) == 0);
  }
  // required for using this class as a key in a set
  bool operator< (const task_identifier &right) const {
    if (!has_name) {
      if (!right.has_name) {
          // if both have an address, return the lower address
          return (address < right.address);
      } else {
          // if left has an address and right doesn't, return true
          return true;
      }
    } else {
      if (right.has_name) {
          // if both have a name, return the lower name
          return (name < right.name);
      }
    }
    // if right has an address and left doesn't, return false
    // (also the default)
    return false;
  }
};

}

/* This is the hash function for the task_identifier class */
namespace std {

  template <>
  struct hash<apex::task_identifier>
  {
    std::size_t operator()(const apex::task_identifier& k) const
    {
      std::size_t h1 = std::hash<std::size_t>()(k.address);
      std::size_t h2 = std::hash<std::string>()(k.name);
      return h1 ^ (h2 << 1);; // instead of boost::hash_combine
    }
  };

}


