/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <vector>

#define APEX_BFD_SYMTAB_LOAD_FAILED        (0)
#define APEX_BFD_SYMTAB_LOAD_SUCCESS        (1)
#define APEX_BFD_SYMTAB_NOT_LOADED        (3)

#define APEX_BFD_NULL_HANDLE                (-1)
#define APEX_BFD_NULL_MODULE_HANDLE        (-1)
#define APEX_BFD_INVALID_MODULE            (-2)

// Iterator function type.  Accepts a symbol address and name.
// The name should be an approximation of the full name, for example,
// the contents of asymbol::name from BFD.  ApexBfd.cpp will
// discover the correct name as needed.
typedef void (*ApexBfdIterFn)(unsigned long, const char *);

typedef int apex_bfd_handle_t;
typedef int apex_bfd_module_handle_t;

struct ApexBfdAddrMap
{
    ApexBfdAddrMap() :
        start(0), end(0), offset(0)
    { }

    ApexBfdAddrMap(unsigned long _start, unsigned long _end,
            unsigned long _offset, char const * _name) :
        start(_start), end(_end), offset(_offset)
    {
        // Safely copy the name string and always
        // end with a NUL char.
        int end = 1;
        if(_name != nullptr) {
            strncpy(name, _name, sizeof(name));
            end = sizeof(name);
        }
        name[end-1] = '\0';
    }

    unsigned long start;
    unsigned long end;
    unsigned long offset;
    char name[512];
};

struct ApexBfdInfo
{
    ApexBfdInfo() :
        probeAddr(0), filename(nullptr), funcname(nullptr), demangled(nullptr),
                lineno(-1), discriminator(0)
    { }
    ~ApexBfdInfo() {
        if (funcname != nullptr && demangled != funcname)
            free(const_cast<char*>(demangled));
    }

    // Makes all fields safe to query
    void secure(unsigned long addr) {
        probeAddr = addr;
        if(!funcname) {
            char * tmp = (char*)malloc(256);
            sprintf(tmp, "addr=<%p>", (void*)(size_t)(addr));
            funcname = tmp;
        }
        if(!filename) filename = "(unknown)";
        if(lineno < 0) lineno = 0;
    }

    unsigned long probeAddr;
    char const * filename;
    char const * funcname;
    char const * demangled;
    int lineno;
        unsigned int discriminator;
};


//
// Main interface functions
//

// Initialize ApexBFD
void Apex_bfd_initializeBfd();

// Register a BFD unit (i.e. an executable and its shared libraries)
apex_bfd_handle_t Apex_bfd_registerUnit();

// free the unit vector
void Apex_delete_bfd_units(void);

// Return true if the given handle is valid
bool Apex_bfd_checkHandle(apex_bfd_handle_t handle);

// Scan the BFD unit for address maps
void Apex_bfd_updateAddressMaps(apex_bfd_handle_t handle);

// Forward lookup of symbol information for a given address.
// Searches the appropriate shared libraries and/or the executable
// for information.
bool Apex_bfd_resolveBfdInfo(apex_bfd_handle_t handle,
        unsigned long probeAddr, ApexBfdInfo & info);

// Fast scan of the executable symbol table.
int Apex_bfd_processBfdExecInfo(apex_bfd_handle_t handle, ApexBfdIterFn fn);

// Fast scan of a module symbol table.
// Note that it's usually not worth doing this since it is unlikely
// that all symbols in the module will be required in a single
// application (e.g. a shared library in a large application).
// Instead, use Apex_bfd_resolveBfdInfo as needed.
int Apex_bfd_processBfdModuleInfo(apex_bfd_handle_t handle,
        apex_bfd_module_handle_t moduleHandle,ApexBfdIterFn fn);

//
// Query functions
//

// Get the address maps in the specified BFD unit
std::vector<ApexBfdAddrMap*> const &
Apex_bfd_getAddressMaps(apex_bfd_handle_t handle);

// Find the address map that probably contains the given address
ApexBfdAddrMap const *
Apex_bfd_getAddressMap(apex_bfd_handle_t handle, unsigned long probeAddr);

// Get the module that possibly defines the given address
apex_bfd_module_handle_t
Apex_bfd_getModuleHandle(apex_bfd_handle_t handle, unsigned long probeAddr);


