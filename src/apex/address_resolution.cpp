//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_bfd.h"
#include <string>
#include <sstream>
#include <iostream>
#include <map>

using namespace std;

namespace apex {

  struct OmpHashNode
  {
    OmpHashNode() { }

    ApexBfdInfo info;        ///< Filename, line number, etc.
    string * location;
  };

  /*
   *-----------------------------------------------------------------------------
   * Simple hash table to map function addresses to region names/identifier
   *-----------------------------------------------------------------------------
   */

  /* destructor helper */
  void delete_hash_table(void);

  /* Define the table of addresses to names */
  struct OmpHashTable : public std::map<uintptr_t, OmpHashNode*>
  {
    OmpHashTable() { }
    virtual ~OmpHashTable() {
      //delete_hash_table();
    }
  };

  /*
   *-----------------------------------------------------------------------------
   * Simple hash table to map function addresses to region names/identifier
   *-----------------------------------------------------------------------------
   */

  /* Static constructor. We only need one. */
  static OmpHashTable & OmpTheHashTable()
  {
    static OmpHashTable htab;
    return htab;
  }

  /* Static BFD unit handle generator. We only need one. */
  static apex_bfd_handle_t & OmpTheBfdUnitHandle()
  {
    static apex_bfd_handle_t OmpbfdUnitHandle = APEX_BFD_NULL_HANDLE;
    if (OmpbfdUnitHandle == APEX_BFD_NULL_HANDLE) {
        OmpbfdUnitHandle = Apex_bfd_registerUnit();
    }
    return OmpbfdUnitHandle;
  }

  /* Delete the hash table. */
  void delete_hash_table(void) {
    // clear the hash map to eliminate memory leaks
    OmpHashTable & mytab = OmpTheHashTable();
    for ( std::map<uintptr_t, OmpHashNode*>::iterator it = mytab.begin(); it != mytab.end(); ++it ) {
      OmpHashNode * node = it->second;
      if (node->location) {
        delete (node->location);
      }
      delete node;
    }
    mytab.clear();
    Apex_delete_bfd_units();
  }

  /* Map a function address to a name and/or source location */
  string * lookup_address(uintptr_t ip, bool withFileInfo) {
    stringstream location;
    apex_bfd_handle_t & OmpbfdUnitHandle = OmpTheBfdUnitHandle();
    OmpHashNode * node = OmpTheHashTable()[ip];
    if (!node) {
      node = new OmpHashNode;
      Apex_bfd_resolveBfdInfo(OmpbfdUnitHandle, ip, node->info);
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
      OmpTheHashTable()[ip] = node;
    }
    if (withFileInfo) {
      return node->location;
    } else {
      return new string(node->info.funcname);
    }
  }
}
