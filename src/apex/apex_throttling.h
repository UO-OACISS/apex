#ifndef APEX_THROTTLING_H
#define APEX_THROTTLING_H

#include "apex_export.h"

extern bool apex_throttleOn;         // Current Throttle status
extern bool apex_checkThrottling;    // Is thread throttling desired
extern bool apex_energyThrottling;   // Try to save power while throttling

#define  RCR_HIGH_POWER_LIMIT  220.0  // system specific cutoff to identify busy systems, WATTS
#define  RCR_LOW_POWER_LIMIT   200.0  // system specific cutoff to identify busy systems, WATTS

#define RCR_MAX_THREADS 48
#define RCR_MIN_THREADS 12

APEX_EXPORT int apex_throttling_policy_periodic(apex_context const context);
APEX_EXPORT int apex_setup_throttling(apex_function_address the_function);      // initialize
APEX_EXPORT int apex_shutdown_throttling(void);   // terminate
APEX_EXPORT int apex_get_thread_cap(void);             // for thread throttling

#endif // APEX_THROTTLING_H
