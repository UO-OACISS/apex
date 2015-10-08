//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "address_resolution.hpp"
#include <sstream>
#include <iostream>

using namespace std;

namespace apex {

  // instantiate our static instance.
  address_resolution * address_resolution::_instance = nullptr;

  /* Map a function address to a name and/or source location */
  string * lookup_address(uintptr_t ip, bool withFileInfo) {
    address_resolution * ar = address_resolution::instance();
    stringstream location;
    address_resolution::my_hash_node * node = ar->my_hash_table[ip];
    if (!node) {
      node = new address_resolution::my_hash_node();
      Apex_bfd_resolveBfdInfo(ar->my_bfd_unit_handle, ip, node->info);
      if (node->info.funcname) {
        location << node->info.funcname ;
      }
      //if (withFileInfo) {
        location << " [{" ;
        if (node->info.filename) {
            location << node->info.filename ;
        }
        location << "} {" << node->info.lineno << ",0}]";
      //}
      node->location = new string(location.str());
      ar->my_hash_table[ip] = node;
    }
    if (withFileInfo) {
      return node->location;
    } else {
      return new string(node->info.funcname);
    }
  }
}
