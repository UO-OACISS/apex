//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "apex_types.h"
#include "utils.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <mutex>
#include <utility>

namespace apex {

class task_identifier {
    public:
    class apex_name_map : public std::unordered_map<std::string, task_identifier*> {
            int tid;
        public:
            apex_name_map();
            ~apex_name_map();
    };

    class apex_addr_map : public std::unordered_map<uint64_t, task_identifier*> {
            int tid;
        public:
            apex_addr_map();
            ~apex_addr_map();
    };

private:
  // some optimizations - since many timers are called over and over, don't
  // create a task ID for every one - use a pool of them.  The problem is
  // that they might have some unique properties - like the data_ptr.  If
  // the data_ptr is used, then we can't "cache" the task ID. But otherwise,
  // this provides a significant speedup, because we don't have to allocate
  // and free lots of tiny objects.
  static apex_name_map& get_task_id_name_map(void);
  static apex_addr_map& get_task_id_addr_map(void);
public:
  apex_function_address address;
  std::string name;
  std::string _resolved_name;
  bool has_name;
  void** _data_ptr; // not included in comparisons!
  bool permanent;
  task_identifier(void) :
      address(0L), name(""), _resolved_name(""), has_name(false), _data_ptr(0), permanent(false) {};
  task_identifier(apex_function_address a, void** data_ptr = 0) :
      address(a), name(""), _resolved_name(""), has_name(false), _data_ptr(data_ptr), permanent(false) {};
  task_identifier(const std::string& n, void** data_ptr = 0) :
      address(0L), name(n), _resolved_name(""), has_name(true), _data_ptr(data_ptr), permanent(false) {
      };
  task_identifier(const task_identifier& rhs) :
      address(rhs.address), name(rhs.name), _resolved_name(rhs._resolved_name), has_name(rhs.has_name), _data_ptr(rhs._data_ptr), permanent(rhs.permanent) { };

  static task_identifier * get_task_id (apex_function_address a, void** data_ptr = 0);
  static task_identifier * get_task_id (const std::string& n, void** data_ptr = 0);
  const std::string& get_name(bool resolve = true);
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


