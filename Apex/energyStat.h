/*
 * starts daemon waittime in nanoseconds 0,000,000. currently ignored
 * and set to .1 sec -- code there but comment out since it was doing 
 * what was expected once in the past
 */
extern "C" int energyDaemonInit(uint64_t waitTime); 

/*
 * prints time since initialization (or last call), energy used, 
 * average power, current tempature of each socket
 */
extern "C" void energyDaemonTerm(); 
