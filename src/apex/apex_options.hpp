#ifndef APEX_OPTIONS_HPP
#define APEX_OPTIONS_HPP

#include <stdlib.h>
#include "apex_types.h"
#include "string.h"
#include "stdio.h"

namespace apex {

class apex_options {
private:
    /* Declare the private member variables */
#define apex_macro(name, member_variable, type, default_value) type _##member_variable;
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    /* Declare the constructor, only used by the "instance" method.
	 * it is defined in the cpp file. */
    apex_options(void);
    /* Disable the copy and assign methods. */
    apex_options(apex_options const&)    = delete;
    void operator=(apex_options const&)  = delete;
public:
    /* The "instance" method. */
    static apex_options& instance(void);
	/* The getter and setter methods */
#define apex_macro(name, member_variable, type, default_value) \
    static void member_variable (type inval); \
    static type member_variable (void);
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
	/* The debugging methods */
    static void print_options(void);
};

}

#endif
