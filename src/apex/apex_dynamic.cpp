#include <dlfcn.h>
#include "apex_dynamic.hpp"
#include "apex_assert.h"
#include "apex_options.hpp"
#include "apex.hpp"
#include <string>
#include <iostream>

namespace apex { namespace dynamic {

void * get_symbol(const char * module, const char * symbol) {
    auto func = dlsym(RTLD_DEFAULT, symbol);
    if (func != nullptr) {
        return func;
    }
    /* If, for some reason, we haven't preloaded the libapex_X.so
     * library, Go looking for the library, and then the symbol.
     * This assumes that the LD_LIBRARY_PATH will include the
     * path to the library. */
    std::string libname{"libapex_"};
    libname += module;
#ifdef __APPLE__
    libname +=".dylib";
#else
    libname +=".so";
#endif
    /* Check to see if we've already loaded it */
    void * handle = dlopen(libname.c_str(),
        RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        /* Library hasn't been loaded, so load it */
        handle = dlopen(libname.c_str(),
            RTLD_NOW | RTLD_LOCAL);
    }
    APEX_ASSERT(handle != nullptr);
    if (handle == nullptr) {
        return nullptr;
        if (apex_options::use_verbose() && apex::instance()->get_node_id() == 0) {
            std::cerr << "Unable to load library " << libname << std::endl;
        }
    }
    func = dlsym(handle, symbol);
    // sanity check
    APEX_ASSERT(func != nullptr);
    if (func == nullptr) {
        if (apex_options::use_verbose() && apex::instance()->get_node_id() == 0) {
            std::cerr << "Unable to load symbol " << symbol << std::endl;
        }
    }
    return func;
}

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
    namespace ompt {
        void apex_ompt_force_shutdown(void);
        typedef void (*apex_ompt_force_shutdown_t)(void);
        void do_shutdown(void) {
            if (apex_options::use_ompt()) {
                // do this once
                static apex_ompt_force_shutdown_t apex_ompt_force_shutdown =
                    (apex_ompt_force_shutdown_t)get_symbol("ompt",
                            "apex_ompt_force_shutdown");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_ompt_force_shutdown != nullptr) {
                    apex_ompt_force_shutdown();
                }
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
            if (apex_options::use_cuda()) {
                // do this once
                static apex_init_cuda_tracing_t apex_init_cuda_tracing =
                    (apex_init_cuda_tracing_t)get_symbol("cuda",
                            "apex_init_cuda_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_init_cuda_tracing != nullptr) {
                    apex_init_cuda_tracing();
                }
            }
        }

        void flush(void) {
            if (apex_options::use_cuda()) {
                // do this once
                static apex_flush_cuda_tracing_t apex_flush_cuda_tracing =
                    (apex_flush_cuda_tracing_t)get_symbol("cuda",
                            "apex_flush_cuda_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_flush_cuda_tracing != nullptr) {
                    apex_flush_cuda_tracing();
                }
            }
        }

        void stop(void) {
            if (apex_options::use_cuda()) {
                // do this once
                static apex_stop_cuda_tracing_t apex_stop_cuda_tracing =
                    (apex_stop_cuda_tracing_t)get_symbol("cuda",
                            "apex_stop_cuda_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_stop_cuda_tracing != nullptr) {
                    apex_stop_cuda_tracing();
                }
            }
        }

    }; // namespace apex::dynamic::cuda

    namespace nvml {
        void apex_nvml_monitor_query(void);
        void apex_nvml_monitor_stop(void);
        double apex_nvml_monitor_getAvailableMemory(void);
        typedef void (*apex_nvml_monitor_query_t)(void);
        typedef void (*apex_nvml_monitor_stop_t)(void);
        typedef double (*apex_nvml_monitor_getAvailableMemory_t)(void);
        void query(void) {
#ifdef APEX_WITH_CUDA
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_nvml_monitor_query_t apex_nvml_monitor_query =
                    (apex_nvml_monitor_query_t)get_symbol("cuda",
                            "apex_nvml_monitor_query");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_nvml_monitor_query != nullptr) {
                    apex_nvml_monitor_query();
                }
            }
#endif
        }
        void stop(void) {
#ifdef APEX_WITH_CUDA
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_nvml_monitor_stop_t apex_nvml_monitor_stop =
                    (apex_nvml_monitor_stop_t)get_symbol("cuda",
                            "apex_nvml_monitor_stop");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_nvml_monitor_stop != nullptr) {
                    apex_nvml_monitor_stop();
                }
            }
#endif
        }
    }

    /* HIP will need several functions. */

    namespace rsmi {
        void apex_rsmi_monitor_query(void);
        void apex_rsmi_monitor_stop(void);
        double apex_rsmi_monitor_getAvailableMemory(void);
        typedef void (*apex_rsmi_monitor_query_t)(void);
        typedef void (*apex_rsmi_monitor_stop_t)(void);
        typedef double (*apex_rsmi_monitor_getAvailableMemory_t)(void);
        void query(void) {
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rsmi_monitor_query_t apex_rsmi_monitor_query =
                    (apex_rsmi_monitor_query_t)get_symbol("hip",
                            "apex_rsmi_monitor_query");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_rsmi_monitor_query != nullptr) {
                    apex_rsmi_monitor_query();
                }
            }
#endif
        }
        void stop(void) {
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rsmi_monitor_stop_t apex_rsmi_monitor_stop =
                    (apex_rsmi_monitor_stop_t)get_symbol("hip",
                            "apex_rsmi_monitor_stop");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_rsmi_monitor_stop != nullptr) {
                    apex_rsmi_monitor_stop();
                }
            }
#endif
        }
        double getAvailableMemory(void) {
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rsmi_monitor_getAvailableMemory_t apex_rsmi_monitor_getAvailableMemory =
                    (apex_rsmi_monitor_getAvailableMemory_t)get_symbol("hip",
                            "apex_rsmi_monitor_getAvailableMemory");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_rsmi_monitor_getAvailableMemory != nullptr) {
                    return apex_rsmi_monitor_getAvailableMemory();
                }
            }
#endif
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
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rocprofiler_monitor_query_t apex_rocprofiler_monitor_query =
                    (apex_rocprofiler_monitor_query_t)get_symbol("hip",
                            "apex_rocprofiler_monitor_query");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_rocprofiler_monitor_query != nullptr) {
                    apex_rocprofiler_monitor_query();
                }
            }
#endif
        }
        void stop(void) {
#ifdef APEX_WITH_HIP
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rocprofiler_monitor_stop_t apex_rocprofiler_monitor_stop =
                    (apex_rocprofiler_monitor_stop_t)get_symbol("hip",
                            "apex_rocprofiler_monitor_stop");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_rocprofiler_monitor_stop != nullptr) {
                    apex_rocprofiler_monitor_stop();
                }
            }
#endif
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
            if (apex_options::use_hip()) {
                // do this once
                static apex_init_hip_tracing_t apex_init_hip_tracing =
                    (apex_init_hip_tracing_t)get_symbol("hip",
                            "apex_init_hip_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_init_hip_tracing != nullptr) {
                    apex_init_hip_tracing();
                }
            }
        }

        void flush(void) {
            if (apex_options::use_hip()) {
                // do this once
                static apex_flush_hip_tracing_t apex_flush_hip_tracing =
                    (apex_flush_hip_tracing_t)get_symbol("hip",
                            "apex_flush_hip_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_flush_hip_tracing != nullptr) {
                    apex_flush_hip_tracing();
                }
            }
        }

        void stop(void) {
            if (apex_options::use_hip()) {
                // do this once
                static apex_stop_hip_tracing_t apex_stop_hip_tracing =
                    (apex_stop_hip_tracing_t)get_symbol("hip",
                            "apex_stop_hip_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_stop_hip_tracing != nullptr) {
                    apex_stop_hip_tracing();
                }
            }
        }

    }; // namespace apex::dynamic::roctracer

        namespace level0 {
            void apex_init_level0_tracing(void);
            void apex_flush_level0_tracing(void);
            void apex_stop_level0_tracing(void);
            typedef void (*apex_init_level0_tracing_t)(void);
            typedef void (*apex_flush_level0_tracing_t)(void);
            typedef void (*apex_stop_level0_tracing_t)(void);
            void init(void) {
                if (apex_options::use_level0()) {
                    // do this once
                    static apex_init_level0_tracing_t apex_init_level0_tracing =
                        (apex_init_level0_tracing_t)get_symbol("level0",
                                "apex_init_level0_tracing");
                    // shouldn't be necessary,
                    // but the assertion doesn't happen with release builds
                    if (apex_init_level0_tracing != nullptr) {
                        apex_init_level0_tracing();
                    }
                }
            }

            void flush(void) {
                if (apex_options::use_level0()) {
                    // do this once
                    static apex_flush_level0_tracing_t apex_flush_level0_tracing =
                        (apex_flush_level0_tracing_t)get_symbol("level0",
                                "apex_flush_level0_tracing");
                    // shouldn't be necessary,
                    // but the assertion doesn't happen with release builds
                    if (apex_flush_level0_tracing != nullptr) {
                        apex_flush_level0_tracing();
                    }
                }
            }

            void stop(void) {
                if (apex_options::use_level0()) {
                    // do this once
                    static apex_stop_level0_tracing_t apex_stop_level0_tracing =
                        (apex_stop_level0_tracing_t)get_symbol("level0",
                                "apex_stop_level0_tracing");
                    // shouldn't be necessary,
                    // but the assertion doesn't happen with release builds
                    if (apex_stop_level0_tracing != nullptr) {
                        apex_stop_level0_tracing();
                    }
                }
            }

        }; // namespace apex::dynamic::level0

    }; }; // namespace apex::dynamic


