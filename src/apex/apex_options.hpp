#ifndef APEX_OPTIONS_HPP
#define APEX_OPTIONS_HPP

#include <stdlib.h>
#include "apex_types.h"
#include "string.h"
#include "stdio.h"

namespace apex {

class apex_options {
private:
#define apex_macro(name, member_variable, type, default_value) type _##member_variable;
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
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
    static apex_options * _instance;
public:
        ~apex_options(void) {};
    static apex_options * instance(void) {
        if (_instance == NULL) { _instance = new apex_options(); }
        return _instance;
    }
#define apex_macro(name, member_variable, type, default_value) \
    static void member_variable (type inval) { instance()->_##member_variable = inval; } \
    static type member_variable (void) { return instance()->_##member_variable; }
    FOREACH_APEX_OPTION(apex_macro)
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
};

}

#endif
