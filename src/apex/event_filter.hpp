/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "apex_options.hpp"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <regex>
#include <vector>

namespace apex {

class event_filter {
public:
    bool exclude(const std::string &name);
    bool have_filter;
    std::vector<std::regex> exclude_names;
    std::vector<std::regex> include_names;
    bool have_include_names;
    /* Declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    event_filter(std::string filename);
    ~event_filter(void) {};
private:
    rapidjson::Document configuration;
};

}

