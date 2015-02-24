#ifndef APEX_THROTTLING_H
#define APEX_THROTTLING_H

#include "apex_export.h"

extern bool apex_throttleOn;         // Current Throttle status
extern bool apex_checkThrottling;    // Is thread throttling desired
extern bool apex_energyThrottling;   // Try to save power while throttling

typedef enum {MAXIMIZE_THROUGHPUT,
	          MAXIMIZE_ACCUMULATED, 
	          MINIMIZE_ACCUMULATED
} apex_optimization_criteria_t;

#define  RCR_HIGH_POWER_LIMIT  220.0  // system specific cutoff to identify busy systems, WATTS
#define  RCR_LOW_POWER_LIMIT   200.0  // system specific cutoff to identify busy systems, WATTS

#define RCR_MAX_THREADS 48
#define RCR_MIN_THREADS 12

APEX_EXPORT int apex_setup_power_cap_throttling(void);      // initialize
APEX_EXPORT int apex_setup_address_throttling(apex_function_address the_address,
        apex_optimization_criteria_t criteria);      // initialize
APEX_EXPORT int apex_setup_name_throttling(const char * the_name,
        apex_optimization_criteria_t criteria);      // initialize
APEX_EXPORT int apex_shutdown_throttling(void);   // terminate
APEX_EXPORT int apex_get_thread_cap(void);             // for thread throttling

#endif // APEX_THROTTLING_H
