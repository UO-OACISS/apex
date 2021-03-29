/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "tau_listener.hpp"
#include <iostream>

using namespace std;

#ifdef APEX_USE_WEAK_SYMBOLS

/* Weak symbols that are redefined if we use TAU at runtime */
extern "C" {
APEX_EXPORT APEX_WEAK_PRE int Tau_init(int, char**) APEX_WEAK_POST {
    /* Print an error message, because TAU wasn't preloaded! */
    std::cerr <<
        "WARNING! TAU libraries not loaded, TAU support unavailable!"
        << std::endl;
    return 0;
}
APEX_EXPORT APEX_WEAK_PRE int
    Tau_register_thread(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int
    Tau_create_top_level_timer_if_necessary(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_start(const char *) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_stop(const char *) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_exit(const char*) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_dump_prefix(const char*) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_set_node(int) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int
    Tau_profile_exit_all_threads(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_get_thread(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int
    Tau_profile_exit_all_tasks(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int Tau_global_stop(void) APEX_WEAK_POST {return 0;}
APEX_EXPORT APEX_WEAK_PRE int
    Tau_trigger_context_event_thread(char*, double, int) APEX_WEAK_POST {return 0;}
}

#endif // APEX_USE_WEAK_SYMBOLS
