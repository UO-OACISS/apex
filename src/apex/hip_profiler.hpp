/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "rocprofiler.h"
#include <hsa.h>

namespace apex { namespace rocprofiler {

class monitor {
public:
    void query();
    void stop();
private:
    /* declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    monitor (void);
    ~monitor (void);
    /* Disable the copy and assign methods. */
    monitor(monitor const&)    = delete;
    void operator=(monitor const&)  = delete;
    bool enabled;
    // HSA status
    hsa_status_t status;
    // Profiling context
    rocprofiler_t* context;
    // Profiling properties
    rocprofiler_properties_t properties;
    // Profiling feature objects
    unsigned feature_count;
    // there won't be more than 8 counters, but this is easier than a dynamic size.
    rocprofiler_feature_t feature[16];
    unsigned group_n;
}; // class monitor

} // namespace rocm_profiler
} // namespace apex
