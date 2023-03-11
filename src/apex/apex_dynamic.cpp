#include <dlfcn.h>
#include "apex_dynamic.hpp"
#include "apex_assert.h"

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
namespace apex { namespace dynamic { namespace ompt {
    void ompt_force_shutdown(void);
    typedef void (*ompt_force_shutdown_t)(void);
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


