/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* This file contains MPI implementations of communication necessary to support
   OTF2 tracing.  For example, event unification and clock Synchronization.
*/

// only compile this file if we have NO networking support!
#if defined(APEX_HAVE_HPX_CONFIG) || defined(APEX_HAVE_HPX)
#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#endif

#if !defined(HPX_HAVE_NETWORKING) && !defined(APEX_WITH_MPI)

#include "otf2_listener.hpp"

namespace apex {

int64_t otf2_listener::synchronizeClocks(void) {
    return 0;
}

int otf2_listener::getCommRank() {
    return apex::instance()->get_node_id();
}

int otf2_listener::getCommSize() {
    return apex::instance()->get_num_ranks();
}

} // namespace apex

#endif
