//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex_options.hpp"
#include "apex.hpp"
#include <iostream>

namespace apex
{
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
