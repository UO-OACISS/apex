//  Copyright (c) 2014-2018 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* This is annoying and confusing.  We have to set a define so that the
 * HPX config file will be included, which will define APEX_HAVE_HPX
 * for us.  We can't use the same name because then the macro is defined
 * twice.  So, we have a macro to make sure the macro is defined. */
#ifdef APEX_HAVE_HPX_CONFIG
#include <hpx/config.hpp>
#endif

#include "apex_options.hpp"
#include "apex.hpp"
#include "apex_config.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include "utils.hpp"
#include "proc_read.h"

namespace apex
{
    static const std::string config_file_name = "apex.conf";
    apex_options * apex_options::_instance(nullptr);

    apex_options::apex_options(void) {

        std::ifstream conf_file(config_file_name, std::ifstream::in);
        if(conf_file.good()) {
            std::string line;
            while(!conf_file.eof()) {
                conf_file >> line;
                std::vector<std::string> parts;
                split(line, '=', parts);
                if(parts.size() == 2) {
#if defined(_MSC_VER)
                   std::string val(parts[0]);
                   val += "=" + parts[1];
                   _putenv(val.c_str());
#else
                   setenv(parts[0].c_str(), parts[1].c_str(), 0);
#endif
                }
            }
            conf_file.close();
        }

        char* option = nullptr;
// getenv is not thread-safe, but the constructor for this static singleton is.
#define apex_macro(name, member_variable, type, default_value) \
    _##member_variable = default_value; \
    option = getenv(#name); \
    if (option != nullptr) { \
        _##member_variable = (type)(atoi(option)); \
    }
    FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

#define apex_macro(name, member_variable, type, default_value) \
    option = getenv(#name); \
    if (option == nullptr) { \
        _##member_variable = strdup(default_value); \
    } else { \
        _##member_variable = strdup(option); \
    }
    FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro

#define apex_macro(name, member_variable, type, default_value) \
    type _##member_variable; /* declare the local variable */ \
    option = getenv(#name); \
    if (option == nullptr) { \
        _##member_variable = strdup(default_value); \
    } else { \
        _##member_variable = strdup(option); \
    }
    FOREACH_EXTERNAL_STRING_OPTION(apex_macro)
#undef apex_macro

#ifdef APEX_HAVE_ACTIVEHARMONY
    // validate the HARMONY_HOME setting - make sure it is set.
    int rc = setenv("HARMONY_HOME", _activeharmony_root, 0);
    if (rc == -1) {
        std::cerr << "Warning - couldn't set HARMONY_HOME" << std::endl;
    }
#endif

    // free the local varaibles
#define apex_macro(name, member_variable, type, default_value) \
    free (_##member_variable);
    FOREACH_EXTERNAL_STRING_OPTION(apex_macro)
#undef apex_macro
    }

    apex_options::~apex_options(void) {
#define apex_macro(name, member_variable, type, default_value) \
        free (_##member_variable);
        FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
    }

    apex_options& apex_options::instance(void) {
        if (_instance == nullptr) {
            _instance = new apex_options();
        }
        return *_instance;
    }

    void apex_options::delete_instance(void) {
        if (_instance != nullptr) {
            delete(_instance);
            _instance = nullptr;
        }
    }

#define apex_macro(name, member_variable, type, default_value) \
    void apex_options::member_variable (type inval) { \
    instance()._##member_variable = inval; } \
    type apex_options::member_variable (void) { \
    return instance()._##member_variable; }
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
#ifdef APEX_HAVE_PROC
        std::string tmpstr(proc_data_reader::get_command_line());
        if (tmpstr.length() > 0) {
            std::cout << "Command line: " << tmpstr << std::endl;
        }
#endif
        return;
    }

    void apex_options::make_default_config() {
        apex* instance = apex::instance();
        APEX_UNUSED(instance);
        apex_options& options = apex_options::instance();
        std::ofstream conf_file(config_file_name, std::ofstream::out);
        if(conf_file.good()) {
#define apex_macro(name, member_variable, type, default_value) \
            conf_file << #name << "=" << options.member_variable() << std::endl;
            FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro
#define apex_macro(name, member_variable, type, default_value) \
            conf_file << #name << "=" << options.member_variable() << std::endl;
            FOREACH_APEX_STRING_OPTION(apex_macro)
#undef apex_macro
            conf_file.close();
            std::cout << "Default config written to : " << config_file_name << std::endl;
        }
        return;
    }
}

using namespace apex;

extern "C" {

#define apex_macro(name, member_variable, type, default_value) \
    void apex_set_##member_variable (type inval) { \
    apex_options::member_variable(inval); } \
    type apex_get_##member_variable (void) { \
    return apex_options::member_variable(); }
    FOREACH_APEX_OPTION(apex_macro)
#undef apex_macro

}
