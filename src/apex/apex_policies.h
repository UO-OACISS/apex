//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef APEX_POLICIES_H
#define APEX_POLICIES_H

#include "apex_export.h"

extern bool apex_throttleOn;         // Current Throttle status
extern bool apex_checkThrottling;    // Is thread throttling desired
extern bool apex_energyThrottling;   // Try to save power while throttling

typedef enum {
    INITIAL_STATE,
    BASELINE,
    INCREASE,
    DECREASE,
    NO_CHANGE} last_action_t;

// system specific cutoff to identify busy systems, WATTS
#define  APEX_HIGH_POWER_LIMIT  220.0
// system specific cutoff to identify busy systems, WATTS
#define  APEX_LOW_POWER_LIMIT   200.0

#define APEX_MAX_THREADS 24
#define APEX_MIN_THREADS 1
#define MAX_WINDOW_SIZE 3

#endif // APEX_POLICIES_H
