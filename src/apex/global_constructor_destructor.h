//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* Portable global constructor/destructor for GCC and MSVC.
 * I found this code here: http://www.gonwan.com/?p=8 */

#ifndef GLOBAL_CONSTRUCTOR_DESTRUCTOR_H
#define GLOBAL_CONSTRUCTOR_DESTRUCTOR_H

#include <stdlib.h>
#if defined (_MSC_VER)
#if (_MSC_VER >= 1500)
/* Visual Studio 2008 and later have __pragma */
#define HAS_CONSTRUCTORS
#define DEFINE_CONSTRUCTOR(_func) \
    static void _func(void); \
    static int _func ## _wrapper(void) { _func(); return 0; } \
    __pragma(section(".CRT$XCU",read)) \
    __declspec(allocate(".CRT$XCU")) static int\
        (* _array ## _func)(void) = _func ## _wrapper;
#define DEFINE_DESTRUCTOR(_func) \
    static void _func(void); \
    static int _func ## _constructor(void) { atexit (_func); return 0; } \
    __pragma(section(".CRT$XCU",read)) \
    __declspec(allocate(".CRT$XCU")) static int\
        (* _array ## _func)(void) = _func ## _constructor;
#elif (_MSC_VER >= 1400)
/* Visual Studio 2005 */
#define HAS_CONSTRUCTORS
#pragma section(".CRT$XCU",read)
#define DEFINE_CONSTRUCTOR(_func) \
    static void _func(void); \
    static int _func ## _wrapper(void) { _func(); return 0; } \
    __declspec(allocate(".CRT$XCU")) static int\
        (* _array ## _func)(void) = _func ## _wrapper;
#define DEFINE_DESTRUCTOR(_func) \
    static void _func(void); \
    static int _func ## _constructor(void) { atexit (_func); return 0; } \
    __declspec(allocate(".CRT$XCU")) static int\
        (* _array ## _func)(void) = _func ## _constructor;
#else
/* Visual Studio 2003 and early versions should use #pragma code_seg() to
 * define pre/post-main functions. */
#error Pre/Post-main function not supported on your version of Visual Studio.
#endif
#elif (__GNUC__ > 2) || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
#define HAS_CONSTRUCTORS
#define DEFINE_CONSTRUCTOR(_func) static void \
    __attribute__((constructor)) _func (void);
#define DEFINE_DESTRUCTOR(_func) static void \
    __attribute__((destructor)) _func (void);
#else
/* not supported */
#define DEFINE_CONSTRUCTOR(_func)
#define DEFINE_DESTRUCTOR(_func)
#warning "Global constructors and destructors not defined!"
#endif

// #ifdef HAS_CONSTRUCTORS
// DEFINE_CONSTRUCTOR(apex_init)
// DEFINE_DESTRUCTOR(apex_finalize)
// #endif

#endif // GLOBAL_CONSTRUCTOR_DESTRUCTOR_H
