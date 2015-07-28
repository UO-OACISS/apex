//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_options.hpp"
#include <iostream>

namespace apex
{

// Global static pointer used to ensure a single instance of the class.
apex_options* apex_options::_instance = NULL;

    void apex_options::print_options() {
        apex_options * inst = instance();
#define apex_macro(name, member_variable, type, default_value) \
        std::cout << #name << " : " << inst->member_variable() << std::endl;
        FOREACH_APEX_OPTION(apex_macro)
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

    void apex_print_options() {
        apex_options::print_options();
        return;
    }

}
