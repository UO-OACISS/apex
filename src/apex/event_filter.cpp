/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include "event_filter.hpp"
#include <regex>
#include <iostream>
#include <rapidjson/istreamwrapper.h>

namespace apex {

void handle_error(std::regex_error& e) {
    switch (e.code()) {
        case std::regex_constants::error_collate:
            std::cerr << "collate" << std::endl;
            break;
        case std::regex_constants::error_ctype:
            std::cerr << "ctype" << std::endl;
            break;
        case std::regex_constants::error_escape:
            std::cerr << "escape" << std::endl;
            break;
        case std::regex_constants::error_backref:
            std::cerr << "backref" << std::endl;
            break;
        case std::regex_constants::error_brack:
            std::cerr << "brack" << std::endl;
            break;
        case std::regex_constants::error_paren:
            std::cerr << "paren" << std::endl;
            break;
        case std::regex_constants::error_brace:
            std::cerr << "brace" << std::endl;
            break;
        case std::regex_constants::error_badbrace:
            std::cerr << "badbrace" << std::endl;
            break;
        case std::regex_constants::error_range:
            std::cerr << "range" << std::endl;
            break;
        case std::regex_constants::error_space:
            std::cerr << "space" << std::endl;
            break;
        case std::regex_constants::error_badrepeat:
            std::cerr << "badrepeat" << std::endl;
            break;
        case std::regex_constants::error_complexity:
            std::cerr << "complexity" << std::endl;
            break;
        case std::regex_constants::error_stack:
            std::cerr << "stack" << std::endl;
            break;
        default:
            std::cerr << "unknown" << std::endl;
            break;
    }
}

event_filter::event_filter(std::string filename) : have_filter(false) {
    try {
        std::ifstream cfg(filename);
        if (!cfg.good()) {
            // fail silently, nothing to do but use defaults
            return;
        }
        rapidjson::IStreamWrapper file_wrapper(cfg);
        configuration.ParseStream(file_wrapper);
        cfg.close();
    } catch (...) {
        // fail silently, nothing to do but use defaults
        return;
    }
    have_filter = true;
    // check if this timer should be explicitly ignored
    if (configuration.HasMember("exclude")) {
        auto & exclude_filter = configuration["exclude"];
        for(auto itr = exclude_filter.Begin(); itr != exclude_filter.End(); ++itr) {
            std::string needle(itr->GetString());
            needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
            try {
                exclude_names.push_back(std::regex(needle));
            } catch (std::regex_error& e) {
                std::cerr << "Error: '" << e.what() << "' in regular expression: "
                          << needle << std::endl;
                handle_error(e);
            }
        }
    }
    // check if this timer should be implicitly ignored
    if (configuration.HasMember("include")) {
        have_include_names = true;
        auto & include_filter = configuration["include"];
        for(auto itr = include_filter.Begin(); itr != include_filter.End(); ++itr) {
            std::string needle(itr->GetString());
            needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
            try {
                include_names.push_back(std::regex(needle));
            } catch (std::regex_error& e) {
                std::cerr << "Error: '" << e.what() << "' in regular expression: "
                          << needle << std::endl;
                handle_error(e);
            }
        }
    }
}

bool event_filter::exclude(const std::string &name) {
    // check if this timer should be explicitly ignored
    for(auto re : exclude_names) {
        if (std::regex_search(name, re)) {
            return true;
        }
    }
    // not found in the exclude filters
    // ...but don't assume anything yet - check for include list
    // check if this timer should be implicitly ignored
    if (have_include_names) {
        for(auto re : include_names) {
            if (std::regex_search(name, re)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

} // namespace apex
