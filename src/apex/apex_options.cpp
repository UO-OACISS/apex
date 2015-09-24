//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_options.hpp"
#include "apex.hpp"
#include <iostream>

namespace apex
{
    apex_options::apex_options(void) {
        char* option = NULL;
// getenv is not thread-safe, but the constructor for this static singleton is.
#define apex_macro(name, member_variable, type, default_value) \
    _##member_variable = default_value; \
    option = getenv(#name); \
    if (option != NULL) { \
        _##member_variable = (type)(atoi(option)); \
    }
    FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#define apex_macro(name, member_variable, type, default_value) \
    option = getenv(#name); \
    if (option == NULL) { \
        int length = strlen(default_value) + 1; \
        _##member_variable = (type)(calloc(length, sizeof(char))); \
        strncpy(_##member_variable, default_value, length); \
    } else { \
        int length = strlen(option) + 1; \
        _##member_variable = (type)(calloc(length, sizeof(char))); \
        strncpy(_##member_variable, option, length); \
    }
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    };

    apex_options& apex_options::instance(void) {
        static apex_options _instance;
        return _instance;
    }

#define apex_macro(name, member_variable, type, default_value) \
    void apex_options::member_variable (type inval) { instance()._##member_variable = inval; } \
    type apex_options::member_variable (void) { return instance()._##member_variable; }
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro

    void apex_options::print_options() {
        apex* instance = apex::instance();
        if (instance->get_node_id() != 0) { return; }
        apex_options& options = apex_options::instance();
#define apex_macro(name, member_variable, type, default_value) \
        std::cout << #name << " : " << options.member_variable() << std::endl;
        FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro
#define apex_macro(name, member_variable, type, default_value) \
        std::cout << #name << " : " << options.member_variable() << std::endl;
        FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
        return;
    }
}

using namespace apex;

extern "C" {

#define apex_macro(name, member_variable, type, default_value) \
    void apex_set_##member_variable (type inval) { apex_options::member_variable(inval); } \
    type apex_get_##member_variable (void) { return apex_options::member_variable(); }
    FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

}
