#include <dlfcn.h>
#include "apex_dynamic.hpp"
#include "apex_assert.h"

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
namespace apex { namespace dynamic {
namespace ompt {
    void apex_ompt_force_shutdown(void);
    typedef void (*apex_ompt_force_shutdown_t)(void);
    void do_shutdown(void) {
        // do this once
        static apex_ompt_force_shutdown_t apex_ompt_force_shutdown =
            (apex_ompt_force_shutdown_t)dlsym(RTLD_DEFAULT,
                "apex_ompt_force_shutdown");
        // sanity check
        APEX_ASSERT(apex_ompt_force_shutdown != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_ompt_force_shutdown != nullptr) {
            apex_ompt_force_shutdown();
        }
    }

}; // namespace apex::dynamic::ompt

/* CUDA will need a few functions. */

namespace cuda {
    void apex_init_cuda_tracing(void);
    void apex_flush_cuda_tracing(void);
    void apex_stop_cuda_tracing(void);
    typedef void (*apex_init_cuda_tracing_t)(void);
    typedef void (*apex_flush_cuda_tracing_t)(void);
    typedef void (*apex_stop_cuda_tracing_t)(void);
    void init(void) {
        // do this once
        static apex_init_cuda_tracing_t apex_init_cuda_tracing =
            (apex_init_cuda_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_init_cuda_tracing");
        // sanity check
        APEX_ASSERT(apex_init_cuda_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_init_cuda_tracing != nullptr) {
            apex_init_cuda_tracing();
        }
    }

    void flush(void) {
        // do this once
        static apex_flush_cuda_tracing_t apex_flush_cuda_tracing =
            (apex_flush_cuda_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_flush_cuda_tracing");
        // sanity check
        APEX_ASSERT(apex_flush_cuda_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_flush_cuda_tracing != nullptr) {
            apex_flush_cuda_tracing();
        }
    }

    void stop(void) {
        // do this once
        static apex_stop_cuda_tracing_t apex_stop_cuda_tracing =
            (apex_stop_cuda_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_stop_cuda_tracing");
        // sanity check
        APEX_ASSERT(apex_stop_cuda_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_stop_cuda_tracing != nullptr) {
            apex_stop_cuda_tracing();
        }
    }

}; // namespace apex::dynamic::cuda

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
    void apex_init_hip_tracing(void);
    void apex_flush_hip_tracing(void);
    void apex_stop_hip_tracing(void);
    typedef void (*apex_init_hip_tracing_t)(void);
    typedef void (*apex_flush_hip_tracing_t)(void);
    typedef void (*apex_stop_hip_tracing_t)(void);
    void init(void) {
        // do this once
        static apex_init_hip_tracing_t apex_init_hip_tracing =
            (apex_init_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_apex_init_hip_tracing");
        // sanity check
        APEX_ASSERT(apex_init_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_init_hip_tracing != nullptr) {
            apex_init_hip_tracing();
        }
    }

    void flush(void) {
        // do this once
        static apex_flush_hip_tracing_t apex_flush_hip_tracing =
            (apex_flush_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_flush_hip_tracing");
        // sanity check
        APEX_ASSERT(apex_flush_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_flush_hip_tracing != nullptr) {
            apex_flush_hip_tracing();
        }
    }

    void stop(void) {
        // do this once
        static apex_stop_hip_tracing_t apex_stop_hip_tracing =
            (apex_stop_hip_tracing_t)dlsym(RTLD_DEFAULT,
                "apex_stop_hip_tracing");
        // sanity check
        APEX_ASSERT(apex_stop_hip_tracing != nullptr);
        // shouldn't be necessary,
        // but the assertion doesn't happen with release builds
        if (apex_stop_hip_tracing != nullptr) {
            apex_stop_hip_tracing();
        }
    }

}; // namespace apex::dynamic::roctracer

}; }; // namespace apex::dynamic


