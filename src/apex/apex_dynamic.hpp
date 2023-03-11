#pragma once

#include <dlfcn.h>
#include "apex_assert.h"

void ompt_force_shutdown(void);
typedef void (*ompt_force_shutdown_t)(void);

namespace apex { namespace dynamic { namespace ompt {
    void do_shutdown(void) {
        // do this once
        static ompt_force_shutdown_t ompt_force_shutdown =
            (ompt_force_shutdown_t)dlsym(RTLD_DEFAULT,
                "ompt_force_shutdown");
        // sanity check
        APEX_ASSERT(ompt_force_shutdown != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (ompt_force_shutdown != nullptr) {
            ompt_force_shutdown();
        }
    }

}; }; }; // namespace apex::dynamic::ompt


