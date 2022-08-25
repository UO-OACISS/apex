/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <stdlib.h>
#include "string.h"
#include "stdio.h"
#include "apex_types.h"
#include "apex_export.h"
#include <atomic>

#define FOREACH_EXTERNAL_STRING_OPTION(macro) \
        macro (HARMONY_HOME, activeharmony_root, char*, ACTIVEHARMONY_ROOT, "") \

namespace apex {

class apex_options {
private:
    // singleton instance.
    static apex_options * _instance;
    /* Declare the private member variables */
#define apex_macro(name, member_variable, type, default_value, description)\
    std::atomic<type> _##member_variable;
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_FLOAT_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    /* Declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    apex_options(void);
    ~apex_options(void);
    /* Disable the copy and assign methods. */
    apex_options(apex_options const&)    = delete;
    void operator=(apex_options const&)  = delete;
public:
    /* The "instance" method. */
    static apex_options& instance(void);
    /* and a cleanup method */
    static void delete_instance(void);
    /* The getter and setter methods */
#define apex_macro(name, member_variable, type, default_value, description) \
    APEX_EXPORT static void member_variable (type inval); \
    APEX_EXPORT static type member_variable (void);
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_FLOAT_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    /* The debugging methods */
    APEX_EXPORT static void print_options(void);
    APEX_EXPORT static void make_default_config(void);
    APEX_EXPORT static void environment_help(void);
};

}

