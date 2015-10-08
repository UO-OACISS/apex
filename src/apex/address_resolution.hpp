//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

#include "apex.hpp"
#include "apex_bfd.h"
#include <string>
#include <map>
#include <mutex>

namespace apex {

  class address_resolution {
      private:
        static address_resolution * _instance;
        address_resolution(void) { 
          my_bfd_unit_handle = Apex_bfd_registerUnit();
        };
        address_resolution(address_resolution const&);  // copy constructor is private
        address_resolution& operator=(address_resolution const& a); // assignment operator is private
      public:
        static std::mutex _bfd_mutex;

      struct my_hash_node
      {
        my_hash_node() { }
        ApexBfdInfo info;        ///< Filename, line number, etc.
        std::string * location;
      };

      static address_resolution * instance() { 
          if (_instance == nullptr) {
              // only one thread should instantiate it!
              std::lock_guard<std::mutex> lock(_bfd_mutex);
              if (_instance == nullptr) {
                  _instance = new address_resolution();
              }
          }
          return _instance; 
      }
      ~address_resolution(void) {
        // call apex::finalize() just in case!
        finalize();
        for ( std::unordered_map<uintptr_t, 
              my_hash_node*>::iterator it = my_hash_table.begin(); 
              it != my_hash_table.end(); ++it ) {
          my_hash_node * node = it->second;
          if (node->location) {
            delete (node->location);
          }
          delete node;
        }
        my_hash_table.clear();
        Apex_delete_bfd_units();
      }
      std::unordered_map<uintptr_t, my_hash_node*> my_hash_table;
      apex_bfd_handle_t my_bfd_unit_handle;
  };

  std::string * lookup_address(uintptr_t ip, bool withFileInfo);

}

