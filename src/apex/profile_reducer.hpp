/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include "apex_types.h"
#include "profiler_listener.hpp"

namespace apex {

std::map<std::string, apex_profile*> reduce_profiles_for_screen();

void reduce_profiles(std::stringstream& header,
    std::stringstream& csv_output, std::string filename, bool flat);
void reduce_flat_profiles(int node_id, int num_papi_counters,
    std::vector<std::string> metric_names,
    profiler_listener* listener);

}
