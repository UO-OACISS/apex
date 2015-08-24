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
    /* Declare the constructor, only used by the "instance" method. */
    apex_options(void) {
        char* option = NULL;
// FIXME: getenv is not thread-safe
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
    /* Disable the copy and assign methods. */
    apex_options(apex_options const&)    = delete;
    void operator=(apex_options const&)  = delete;
public:
    /* The "instance" method. */
    static apex_options& instance(void) {
    	static apex_options _instance;
        return _instance;
    }
#define apex_macro(name, member_variable, type, default_value) \
    static void member_variable (type inval) { instance()._##member_variable = inval; } \
    static type member_variable (void) { return instance()._##member_variable; }
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    static void print_options(void);
};

}

#endif
