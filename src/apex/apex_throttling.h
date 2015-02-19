#include "libenergy.h"

extern bool apex_throttleOn;         // Current Throttle status
extern bool apex_checkThrottling;    // Is thread throttling desired
extern bool apex_energyThrottling;   // Try to save power while throttling

#define  RCR_HIGH_POWER_LIMIT  220.0  // system specific cutoff to identify busy systems, WATTS
#define  RCR_LOW_POWER_LIMIT   200.0  // system specific cutoff to identify busy systems, WATTS

#define RCR_MAX_THREADS 48
#define RCR_MIN_THREADS 12

int apex_throttling_policy_periodic(apex_context const context);
int apex_setup_throttling(void);      // initialize
int apex_shutdown_throttling(void);   // terminate
int apex_get_thread_cap(void);             // for thread throttling

// RCR functions
void energyDaemonInit(void);  // function in RCR to setup energy counters
void energyDaemonTerm(void);  // function in RCR to print final energy usage
void energyDaemonEnter(void); // function in RCR to start energy collection
//void setDutyCycle(char option, int rank, int value);  // added akp
