#ifndef APEX_THROTTLING_H
#define APEX_THROTTLING_H

#include "apex_export.h"

extern bool apex_throttleOn;         // Current Throttle status
extern bool apex_checkThrottling;    // Is thread throttling desired
extern bool apex_energyThrottling;   // Try to save power while throttling

#define  APEX_HIGH_POWER_LIMIT  220.0  // system specific cutoff to identify busy systems, WATTS
#define  APEX_LOW_POWER_LIMIT   200.0  // system specific cutoff to identify busy systems, WATTS

#define APEX_MAX_THREADS 48
#define APEX_MIN_THREADS 12

#endif // APEX_THROTTLING_H