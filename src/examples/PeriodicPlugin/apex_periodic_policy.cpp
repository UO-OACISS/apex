//  APEX OpenMP Policy
//
//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <stdio.h>
#include "apex_api.hpp"
#include "apex_policies.hpp"

static apex_policy_handle * start_policy_handle{nullptr};
bool apex_policy_running{false};

int periodic_policy(const apex_context context) {
	std::cout << __func__ << std::endl;
    if(context.data == nullptr) {
        std::cerr << "ERROR: No task_identifier for event!" << std::endl;
        return APEX_ERROR;
    }
    auto task_ids = apex::get_available_profiles();
    std::stringstream ss;
    ss << "Found " << task_ids.size() << " profiles so far.\n";
    for (auto t : task_ids) {
        ss << t.get_name() << "\n";
    }
    /* Get one specific task_identifier */
    apex::task_identifier tid("pthread_join");
    apex_profile * profile = get_profile(tid);
    if (profile) {
        ss << "pthread_join : Num Calls : " << profile->calls << "\n";
        ss << "pthread_join : Accumulated : " << profile->accumulated << "\n";
        ss << "pthread_join : Max : " << profile->maximum << "\n";
    }
    std::cout << ss.str() << std::endl;

    return APEX_NOERROR;
}

int register_policy() {
    // Register the policy functions with APEX
    std::function<int(apex_context const&)> periodic_policy_fn{periodic_policy};
    start_policy_handle = apex::register_periodic_policy(1000000, periodic_policy_fn);
    if(start_policy_handle == nullptr) {
        return APEX_ERROR;
    }
    return APEX_NOERROR;
}

extern "C" {

    int apex_plugin_init() {
		std::cout << __func__ << std::endl;
        if(!apex_policy_running) {
            fprintf(stderr, "apex_openmp_policy init\n");
            int status = register_policy();
            apex_policy_running = true;
            return status;
        } else {
            fprintf(stderr, "Unable to start apex_openmp_policy because it is already running.\n");
            return APEX_ERROR;
        }
    }

    int apex_plugin_finalize() {
		std::cout << __func__ << std::endl;
        if(apex_policy_running) {
            fprintf(stderr, "apex_openmp_policy finalize\n");
            apex::deregister_policy(start_policy_handle);
            apex_policy_running = false;
            return APEX_NOERROR;
        } else {
            fprintf(stderr, "Unable to stop apex_openmp_policy because it is not running.\n");
            return APEX_ERROR;
        }
    }

}

