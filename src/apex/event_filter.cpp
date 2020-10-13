
#include "event_filter.hpp"
#include <regex>
#include <iostream>
#include <rapidjson/istreamwrapper.h>

namespace apex {

event_filter * event_filter::_instance{nullptr};

event_filter::event_filter() {
    try {
        std::ifstream cfg(apex_options::task_event_filter_file());
        if (!cfg.good()) {
            // fail silently, nothing to do but use defaults
            no_filter = true;
            return;
        }
        rapidjson::IStreamWrapper file_wrapper(cfg);
        configuration.ParseStream(file_wrapper);
        cfg.close();
        no_filter = false;
    } catch (...) {
        no_filter = true;
    }
}

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

bool event_filter::_include(const std::string &name) {
    if (no_filter) { return true; }
    if (configuration.HasMember("include")) {
        auto & include_filter = configuration["include"];
        for(auto itr = include_filter.Begin(); itr != include_filter.End(); ++itr) {
            std::string needle(itr->GetString());
            needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
            try {
                std::regex re(needle);
                std::string haystack(name);
                if (std::regex_search(haystack, re)) {
                    return true;
                }
            } catch (std::regex_error& e) {
                std::cerr << "Error: '" << e.what() << "' in regular expression: "
                          << needle << std::endl;
                handle_error(e);
            }
        }
        // not found in the include filters
        return false;
    }
    if (configuration.HasMember("exclude")) {
        auto & exclude_filter = configuration["exclude"];
        for(auto itr = exclude_filter.Begin(); itr != exclude_filter.End(); ++itr) {
            std::string needle(itr->GetString());
            needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
            try {
                std::regex re(needle);
                std::string haystack(name);
                if (std::regex_search(haystack, re)) {
                    return false;
                }
            } catch (std::regex_error& e) {
                std::cerr << "Error: '" << e.what() << "' in regular expression: "
                          << needle << std::endl;
                handle_error(e);
            }
        }
        // not found in the exclude filters
        return true;
    }
    return true; // no filters
}

bool event_filter::include(const std::string &name) {
    return instance()._include(name);
}

event_filter& event_filter::instance(void) {
    static std::mutex mtx;
    if (_instance == nullptr) {
        mtx.lock();
        if (_instance == nullptr) {
            _instance = new event_filter();
        }
        mtx.unlock();
    }
    return *_instance;
}


} // namespace apex
