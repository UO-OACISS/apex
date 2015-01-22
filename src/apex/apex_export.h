//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef APEX_EXPORT_H
#define APEX_EXPORT_H

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#define APEX_EXPORT __declspec(dllexport)
#else
#define APEX_EXPORT __attribute__((visibility("default")))
#endif

#endif
