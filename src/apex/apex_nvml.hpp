/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "nvml.h"
#include <set>
#include <mutex>

namespace apex { namespace nvml {

class monitor {
public:
    void query();
    void stop();
    static void activateDeviceIndex(uint32_t index);
    static monitor& instance();
private:
    /* declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    monitor (void);
    ~monitor (void);
    /* Disable the copy and assign methods. */
    monitor(monitor const&)    = delete;
    void operator=(monitor const&)  = delete;
    bool success;
    uint32_t deviceCount;
    std::vector<nvmlDevice_t> devices;
    std::vector<bool> queried_once;
    static std::set<uint32_t> activeDeviceIndices;
    static std::mutex indexMutex;
    double convertValue(nvmlFieldValue_t &value);
}; // class monitor

} // namespace nvml
} // namespace apex
