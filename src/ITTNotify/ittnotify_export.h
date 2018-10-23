//  Copyright (c) 2014-2018 University of Oregon
//  Copyright (c) 2014-2018 Kevin Huck
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ITTNOTIFY_EXPORT_H
#define ITTNOTIFY_EXPORT_H

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#define ITTNOTIFY_EXPORT __declspec(dllexport)
#else
#define ITTNOTIFY_EXPORT __attribute__((visibility("default")))
#endif

#endif
