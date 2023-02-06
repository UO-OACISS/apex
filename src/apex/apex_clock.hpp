/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <chrono>
#if defined(APEX_WITH_LEVEL0) // needed to map to GPU time
#define MYCLOCK std::chrono::steady_clock
#else
#define MYCLOCK std::chrono::system_clock
#endif

namespace apex {

class our_clock {
public:
    // need this before the task_wrapper uses it.
    static uint64_t time_point_to_nanoseconds(std::chrono::time_point<MYCLOCK> tp) {
        auto value = tp.time_since_epoch();
        uint64_t duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
        return duration;
    }
    static uint64_t now_ns() {
        return time_point_to_nanoseconds(MYCLOCK::now());
    }
};

} // namespace

