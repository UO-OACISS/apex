#pragma once

#include "apex_api.hpp"

int apex_example_policy_func(apex_context const context);
void apex_example_set_function_address(apex_function_address addr);
long apex_example_get_active_threads(void);
void apex_example_set_rank_info(int me, int all);
