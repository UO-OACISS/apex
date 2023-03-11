#include <dlfcn.h>
#include "apex_dynamic.hpp"
#include "apex_assert.h"

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
namespace apex { namespace dynamic {
namespace ompt {
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

}; // namespace apex::dynamic::ompt

/* HIP will need several functions. */

namespace rsmi {
    void apex_rsmi_monitor_query(void);
    void apex_rsmi_monitor_stop(void);
    double apex_rsmi_monitor_getAvailableMemory(void);
    typedef void (*apex_rsmi_monitor_query_t)(void);
    typedef void (*apex_rsmi_monitor_stop_t)(void);
    typedef double (*apex_rsmi_monitor_getAvailableMemory_t)(void);
    void query(void) {
        // do this once
        static apex_rsmi_monitor_query_t apex_rsmi_monitor_query =
            (apex_rsmi_monitor_query_t)dlsym(RTLD_DEFAULT,
                "apex_rsmi_monitor_query");
        // sanity check
        APEX_ASSERT(apex_rsmi_monitor_query != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_rsmi_monitor_query != nullptr) {
            apex_rsmi_monitor_query();
        }
    }
    void stop(void) {
        // do this once
        static apex_rsmi_monitor_stop_t apex_rsmi_monitor_stop =
            (apex_rsmi_monitor_stop_t)dlsym(RTLD_DEFAULT,
                "apex_rsmi_monitor_stop");
        // sanity check
        APEX_ASSERT(apex_rsmi_monitor_stop != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_rsmi_monitor_stop != nullptr) {
            apex_rsmi_monitor_stop();
        }
    }
    double getAvailableMemory(void) {
        // do this once
        static apex_rsmi_monitor_getAvailableMemory_t apex_rsmi_monitor_getAvailableMemory =
            (apex_rsmi_monitor_getAvailableMemory_t)dlsym(RTLD_DEFAULT,
                "apex_rsmi_monitor_getAvailableMemory");
        // sanity check
        APEX_ASSERT(apex_rsmi_monitor_getAvailableMemory != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_rsmi_monitor_getAvailableMemory != nullptr) {
            return apex_rsmi_monitor_getAvailableMemory();
        }
        return 0.0;
    }
}

/* Rocprof will need several functions. */

namespace rocprofiler {
    void apex_rocprofiler_monitor_query(void);
    void apex_rocprofiler_monitor_stop(void);
    typedef void (*apex_rocprofiler_monitor_query_t)(void);
    typedef void (*apex_rocprofiler_monitor_stop_t)(void);
    void query(void) {
        // do this once
        static apex_rocprofiler_monitor_query_t apex_rocprofiler_monitor_query =
            (apex_rocprofiler_monitor_query_t)dlsym(RTLD_DEFAULT,
                "apex_rocprofiler_monitor_query");
        // sanity check
        APEX_ASSERT(apex_rocprofiler_monitor_query != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_rocprofiler_monitor_query != nullptr) {
            apex_rocprofiler_monitor_query();
        }
    }
    void stop(void) {
        // do this once
        static apex_rocprofiler_monitor_stop_t apex_rocprofiler_monitor_stop =
            (apex_rocprofiler_monitor_stop_t)dlsym(RTLD_DEFAULT,
                "apex_rocprofiler_monitor_stop");
        // sanity check
        APEX_ASSERT(apex_rocprofiler_monitor_stop != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_rocprofiler_monitor_stop != nullptr) {
            apex_rocprofiler_monitor_stop();
        }
    }
}

namespace roctracer {
    void init_hip_tracing(void);
    void flush_hip_tracing(void);
    void stop_hip_tracing(void);
    typedef void (*init_hip_tracing_t)(void);
    typedef void (*flush_hip_tracing_t)(void);
    typedef void (*stop_hip_tracing_t)(void);
    void init(void) {
        // do this once
        static init_hip_tracing_t init_hip_tracing =
            (init_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "init_hip_tracing");
        // sanity check
        APEX_ASSERT(init_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (init_hip_tracing != nullptr) {
            init_hip_tracing();
        }
    }

    void flush(void) {
        // do this once
        static flush_hip_tracing_t flush_hip_tracing =
            (flush_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "flush_hip_tracing");
        // sanity check
        APEX_ASSERT(flush_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (flush_hip_tracing != nullptr) {
            flush_hip_tracing();
        }
    }

    void stop(void) {
        // do this once
        static stop_hip_tracing_t stop_hip_tracing =
            (stop_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "stop_hip_tracing");
        // sanity check
        APEX_ASSERT(stop_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (stop_hip_tracing != nullptr) {
            stop_hip_tracing();
        }
    }

}; // namespace apex::dynamic::roctracer

}; }; // namespace apex::dynamic


