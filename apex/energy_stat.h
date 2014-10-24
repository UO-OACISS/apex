/*  Copyright (c) 2014 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

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
