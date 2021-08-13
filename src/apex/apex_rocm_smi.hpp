/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "rocm_smi/rocm_smi.h"
#include <set>
#include <mutex>

namespace apex { namespace rsmi {

class monitor {
public:
    monitor (void);
    ~monitor (void);
    void query();
    static void activateDeviceIndex(uint32_t index);
private:
    uint32_t deviceCount;
    std::vector<uint64_t> devices;
    std::vector<bool> queried_once;
    static std::set<uint32_t> activeDeviceIndices;
    static std::mutex indexMutex;
    //double convertValue(nvmlFieldValue_t &value);
}; // class monitor

} // namespace rsmi
} // namespace apex
