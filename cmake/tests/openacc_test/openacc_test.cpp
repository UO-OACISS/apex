/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <sstream>
#include "apex_openacc.hpp"

/* Function called by OpenACC runtime to register callbacks */
extern "C" {

void acc_register_library(acc_prof_reg reg,
    acc_prof_reg unreg, APEX_OPENACC_LOOKUP_FUNC lookup) {
    DEBUG_PRINT("Inside acc_register_library\n");
    APEX_UNUSED(unreg);
    APEX_UNUSED(lookup);

} // acc_register_library

void apex_openacc_launch_callback(acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info) {
}

void apex_openacc_other_callback( acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info ) {
}

void apex_openacc_data_callback(acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info ) {
}


} // extern "C"
