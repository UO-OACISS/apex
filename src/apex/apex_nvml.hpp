/*  Copyright (c) 2020 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "nvml.h"

namespace apex { namespace nvml {

class monitor {
public:
    monitor (void);
    ~monitor (void);
    void query();
private:
    uint32_t deviceCount;
    std::vector<nvmlDevice_t> devices;
}; // class monitor

} // namespace nvml
} // namespace apex
