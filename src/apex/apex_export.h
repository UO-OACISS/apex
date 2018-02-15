//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef APEX_EXPORT_H
#define APEX_EXPORT_H

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#define APEX_EXPORT __declspec(dllexport)
#define APEX_WEAK_PRE __declspec(selectany)
#define APEX_WEAK_POST
#else

#define APEX_EXPORT __attribute__((visibility("default")))

#ifdef __clang__
#define APEX_WEAK_PRE
#define APEX_WEAK_POST __attribute__((weak_import))
#else
#define APEX_WEAK_PRE __attribute__((weak))
#define APEX_WEAK_POST
#endif

#endif

#if defined(_MSC_VER) || defined(__clang__)
#define APEX_TOP_LEVEL_PACKAGE ::apex
#else
#define APEX_TOP_LEVEL_PACKAGE apex
#endif

#endif
