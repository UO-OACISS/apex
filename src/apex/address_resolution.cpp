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
#include <execinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <dlfcn.h>
#define _XOPEN_SOURCE 600 /* Single UNIX Specification, Version 3 */
#if defined(TAU_HAVE_CORESYMBOLICATION)
#include "CoreSymbolication.h"
#endif
#else
// For PIE offset
#include <link.h>
#endif /* __APPLE__ */

using namespace std;

extern char _start;

namespace apex {

  // instantiate our static instances.
  address_resolution * address_resolution::_instance = nullptr;
  shared_mutex_type address_resolution::_bfd_mutex;

  /* Map a function address to a name and/or source location */
  string * lookup_address(uintptr_t ip_in, bool withFileInfo, bool forceSourceInfo) {
    address_resolution * ar = address_resolution::instance();
    static uintptr_t base_addr = ar->getPieOffset();
    uintptr_t ip = ip_in - base_addr;
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
        node->info.filename = nullptr;
        node->info.funcname = nullptr;
        node->info.lineno = 0;
        node->info.demangled = nullptr;
        node->location = nullptr;
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
        // Apple doesn't give us line numbers.
      }
#endif
#else
#ifdef APEX_HAVE_BFD
        Apex_bfd_resolveBfdInfo(ar->my_bfd_unit_handle, ip, node->info);
#else
        void * const buffer[1] = {(void *)ip};
        char ** names = backtrace_symbols((void * const *)buffer, 1);
        /* Split the backtrace strings into tokens, and get the 4th one */
        std::vector<std::string> result;
	    std::istringstream iss(names[0]);
	    for (std::string s; iss >> s; ) {
		    result.push_back(s);
        }
        node->info.probeAddr = ip;
        node->info.filename = strdup("?");
        node->info.funcname = strdup(result[3].c_str());
#endif
#endif
        if (node->info.filename == nullptr) {
            stringstream ss;
            ss << "UNRESOLVED  ADDR 0x" << hex << ip;
            node->info.funcname = strdup(ss.str().c_str());
        }

        if (node->info.demangled) {
            location << node->info.demangled ;
        } else if (node->info.funcname) {
            std::string mangled(node->info.funcname);
            std::string demangled = demangle(mangled);
            location << demangled ;
        }
        if (apex_options::use_source_location() || forceSourceInfo) {
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
    if (node->info.demangled && (strlen(node->info.demangled) == 0)) {
        node->info.demangled = nullptr;
    }
    if (withFileInfo) {
      return node->location;
    } else {
      return new string(node->info.funcname);
    }
  }

    // gives us the -pie offset in the executable.
    uintptr_t address_resolution::getPieOffset() {
#if defined(__APPLE__)
        return 0UL;
#else
#if 1
        uintptr_t entry_point = _r_debug.r_map->l_addr;
        return entry_point;
#else
        Dl_info info;
        void *extra = NULL;
        dladdr1(&_start, &info, &extra, RTLD_DL_LINKMAP);
        struct link_map *map = (struct link_map*)extra;
        return (uintptr_t)(map->l_addr);
#endif
#endif
    }

}
