#pragma once

#include "apex.hpp"
#include "apex_options.hpp"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace apex {

class event_filter {
public:
    static bool exclude(const std::string &name);
    static event_filter& instance(void);
    bool have_filter;
private:
    /* Declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    event_filter(void);
    ~event_filter(void) {};
    /* Disable the copy and assign methods. */
    event_filter(event_filter const&)    = delete;
    void operator=(event_filter const&)  = delete;
    bool _exclude(const std::string &name);
    static event_filter * _instance;
    rapidjson::Document configuration;
};

}

