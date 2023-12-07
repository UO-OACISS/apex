// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Don't use this method if we are using static constructor/destructors
// on platforms like Apple, PPC64le, etc.
#ifndef APEX_USE_STATIC_GLOBAL_CONSTRUCTOR

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include "apex_api.hpp"

#if defined(__GNUC__)
#define __APEX_FUNCTION__ __PRETTY_FUNCTION__
#else
#define __APEX_FUNCTION__ __func__
#endif

extern "C" {

// Trampoline for the real main()
static int (*main_real)(int, char**, char**);

int apex_preload_main(int argc, char** argv, char** envp) {
    // FIRST! check to see if this is a bash script. if so, DO NOTHING
    size_t len{strlen(argv[0])};
    if (len > 4 && strncmp(argv[0] + (len - 4), "bash", 4)) {
        return main_real(argc, argv, envp);
    }
    // FIRST! check to see if this is a [t]csh script. if so, DO NOTHING
    if (len > 3 && strncmp(argv[0] + (len - 3), "csh", 3)) {
        return main_real(argc, argv, envp);
    }
    // FIRST! check to see if this is gdb. if so, DO NOTHING (should get caught by the apex_exec script though)
    if (len > 3 && strncmp(argv[0] + (len - 3), "gdb", 3)) {
        return main_real(argc, argv, envp);
    }
    // prevent re-entry
    static int _reentry = 0;
    if(_reentry > 0) {
        return main_real(argc, argv, envp);
    }
    _reentry = 1;

    int ret{0};
    /* If we're using OTF2, make sure we don't initialize until after MPI_Init */
    if (apex::apex_options::use_otf2() && apex::apex_options::use_mpi()) {
        ret = main_real(argc, argv, envp);
    } else {
        apex::init("APEX Preload", 0, 1);
        auto t = apex::new_task(__APEX_FUNCTION__);
        apex::start(t);
        ret = main_real(argc, argv, envp);
        apex::stop(t);
        apex::finalize();
        if (apex::apex_options::use_screen_output()) {
            apex::dump(false);
        }
        apex::cleanup();
    }
    return ret;
}

typedef int
(*apex_preload_libc_start_main)(int (*)(int, char**, char**), int, char**,
                  int (*)(int, char**, char**), void (*)(void),
                  void (*)(void), void*);

int
__libc_start_main(int (*_main)(int, char**, char**), int _argc, char** _argv,
                  int (*_init)(int, char**, char**), void (*_fini)(void),
                  void (*_rtld_fini)(void), void* _stack_end)
{
    // prevent re-entry
    static int _reentry = 0;
    if(_reentry > 0) return -1;
    _reentry = 1;

    // get the address of this function
    void* _this_func = __builtin_return_address(0);

    // Save the real main function address
    main_real = _main;

    // Find the real __libc_start_main()
    apex_preload_libc_start_main user_main = (apex_preload_libc_start_main)dlsym(RTLD_NEXT, "__libc_start_main");

    if(user_main && (void*)(user_main) != _this_func) {
        return user_main(apex_preload_main, _argc, _argv, _init, _fini, _rtld_fini, _stack_end);
    } else {
        fputs("Error! apex_preload could not find __libc_start_main!", stderr);
        return -1;
    }
}

} // extern "C"

#endif // #ifndef APEX_USE_STATIC_GLOBAL_CONSTRUCTOR
