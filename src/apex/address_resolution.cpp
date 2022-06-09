/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "address_resolution.hpp"
#include <sstream>
#include <iostream>
#include <unordered_map>

#ifdef __APPLE__
#include <dlfcn.h>
#define _XOPEN_SOURCE 600 /* Single UNIX Specification, Version 3 */
#if defined(TAU_HAVE_CORESYMBOLICATION)
#include "CoreSymbolication.h"
#endif
#endif /* __APPLE__ */

using namespace std;

namespace apex {

  // instantiate our static instances.
  address_resolution * address_resolution::_instance = nullptr;
  shared_mutex_type address_resolution::_bfd_mutex;

  /* Map a function address to a name and/or source location */
  string * lookup_address(uintptr_t ip, bool withFileInfo) {
    address_resolution * ar = address_resolution::instance();
    stringstream location;
    address_resolution::my_hash_node * node = nullptr;
    std::unordered_map<uintptr_t, address_resolution::my_hash_node*>::const_iterator it;
    {
        read_lock_type l(ar->_bfd_mutex);
        it = ar->my_hash_table.find(ip);
    }
    // address not found? We need to resolve it.
    if (it == ar->my_hash_table.end()) {
      // only one thread should resolve it.
      write_lock_type l(ar->_bfd_mutex);
      // now that we have the lock, did someone else resolve it?
      const std::unordered_map<uintptr_t,
            address_resolution::my_hash_node*>::const_iterator it2 =
            ar->my_hash_table.find(ip);
      if (it2 == ar->my_hash_table.end()) {
        // ...no - so go get it!
        node = new address_resolution::my_hash_node();
#if defined(__APPLE__)
#if defined(APEX_HAVE_CORESYMBOLICATION)
      static CSSymbolicatorRef symbolicator = CSSymbolicatorCreateWithPid(getpid());
      CSSourceInfoRef source_info = CSSymbolicatorGetSourceInfoWithAddressAtTime(symbolicator, (vm_address_t)ip, kCSNow);
      if(CSIsNull(source_info)) {
      } else {
          CSSymbolRef symbol = CSSourceInfoGetSymbol(source_info);
          node->info.probeAddr = ip;
          node->info.filename = strdup(CSSourceInfoGetPath(source_info));
          node->info.funcname = strdup(CSSymbolGetName(symbol));
          node->info.lineno = CSSourceInfoGetLineNumber(source_info);
      }
      //CSRelease(source_info);
#else
      Dl_info info;
      int rc = dladdr((const void *)ip, &info);
      if (rc == 0) {
      } else {
        node->info.probeAddr = ip;
        node->info.filename = strdup(info.dli_fname);
        node->info.funcname = strdup(info.dli_sname);
        node->info.lineno = 0; // Apple doesn't give us line numbers.
      }
#endif
#else
        Apex_bfd_resolveBfdInfo(ar->my_bfd_unit_handle, ip, node->info);
#endif

        if (node->info.demangled) {
          location << node->info.demangled ;
        } else if (node->info.funcname) {
          location << node->info.funcname ;
        }
        if (apex_options::use_source_location()) {
            location << " [{" ;
            if (node->info.filename) {
            location << node->info.filename ;
            }
            location << "} {";
            if (node->info.lineno != 0) {
                location << node->info.lineno << ",0}]";
            } else {
                location << "0x" << std::hex << ip << "}]";
            }
        } else {
            if (node->info.lineno != 0) {
                // to disambiguate C++ functions
                location << ":" << node->info.lineno;
            } else {
                location << ":0x" << std::hex << ip;
            }
        }
        node->location = new string(location.str());
        ar->my_hash_table[ip] = node;
      } else {
        node = it2->second;
      }
    } else {
      node = it->second;
    }
    if (withFileInfo) {
      return node->location;
    } else {
      if (node->info.funcname != nullptr) {
        return new string(node->info.funcname);
      } else {
        return new string("<unknown>");
      }
    }
  }
}
