//  Copyright (c) 2015 University of Oregon
//

#ifndef APEX_EXPORT_H
#define APEX_EXPORT_H

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#define APEX_EXPORT __declspec(dllexport)
#else
#define APEX_EXPORT __attribute__((visibility("default")))
#endif

#endif
