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
    std::string libname{module};
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
        if (apex_options::use_verbose() && apex::instance()->get_node_id() == 0) {
            std::cerr << "Unable to load library " << libname << std::endl;
        }
        return nullptr;
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
#ifdef __APPLE__
        const char * module ="libapex_ompt.dylib";
#else
        const char * module ="libapex_ompt.so";
#endif
        void apex_ompt_force_shutdown(void);
        typedef void (*apex_ompt_force_shutdown_t)(void);
        void do_shutdown(void) {
            if (apex_options::use_ompt()) {
                // do this once
                static apex_ompt_force_shutdown_t apex_ompt_force_shutdown =
                    (apex_ompt_force_shutdown_t)get_symbol(module,
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
#ifdef __APPLE__
        const char * module ="libapex_cuda.dylib";
#else
        const char * module ="libapex_cuda.so";
#endif
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
                    (apex_init_cuda_tracing_t)get_symbol(module,
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
                    (apex_flush_cuda_tracing_t)get_symbol(module,
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
                    (apex_stop_cuda_tracing_t)get_symbol(module,
                            "apex_stop_cuda_tracing");
                // shouldn't be necessary,
                // but the assertion doesn't happen with release builds
                if (apex_stop_cuda_tracing != nullptr) {
                    apex_stop_cuda_tracing();
                }
            }
        }

    }; // namespace apex::dynamic::cuda

    namespace nvtx {
#ifdef __APPLE__
        const char * module =apex_options::nvtx_library();
#else
        const char * module =apex_options::nvtx_library();
#endif
        typedef int (*apex_nvtx_range_push_t)(const char *);
        typedef int (*apex_nvtx_range_pop_t)(void);
        const uint32_t colors[] = { 0xff008800, 0xff000088, 0xff888800, 0xff880088, 0xff008888, 0xff880000, 0xff888888,
                                                            0xff88ff00, 0xff8800ff, 0xff0088ff,
                                                            0xffff8800, 0xffff0088, 0xff00ff88,
                                                                                                            0xffff8888,
                                                                                                            0xffffff88,
                                                                                                            0xff88ff88,
                                                                                                            0xff88ffff,
                                                                                                            0xff8888ff,
                                    0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
        const uint32_t num_colors = 25;
        uint32_t getColor(const char * message) {
            static std::unordered_map<std::string, uint32_t> theMap;
            std::string tmp{message};
            if (theMap.count(tmp) == 0) {
                size_t index = theMap.size();
                uint32_t color_index = index % num_colors;
                theMap.insert(std::pair<std::string, uint32_t>(tmp, color_index));
                return color_index;
            }
            return theMap[tmp];
        }
        // these values are from nvToolsExt.h, cuda 12.3.2
        const int16_t NVTX_VERSION = 2;
        const int32_t NVTX_COLOR_ARGB = 1;
        const int32_t NVTX_MESSAGE_TYPE_ASCII = 1;
        typedef union nvtxMessageValue_t {
            const char* ascii;
            const wchar_t* unicode;
            /* NVTX_VERSION_2 */
            // nvtxStringHandle_t registered;
        } nvtxMessageValue_t;
        typedef struct nvtxEventAttributes_v2 {
            uint16_t version;
            uint16_t size;
            uint32_t category;
            int32_t colorType;              /* nvtxColorType_t */
            uint32_t color;
            int32_t payloadType;            /* nvtxPayloadType_t */
            int32_t reserved0;
            union payload_t {
                uint64_t ullValue;
                int64_t llValue;
                double dValue;
                /* NVTX_VERSION_2 */
                uint32_t uiValue;
                int32_t iValue;
                float fValue;
            } payload;
            int32_t messageType;            /* nvtxMessageType_t */
            nvtxMessageValue_t message;
        } nvtxEventAttributes_v2;
        typedef struct nvtxEventAttributes_v2 nvtxEventAttributes_t;
        typedef int (*apex_nvtx_range_push_ex_t)(const nvtxEventAttributes_t* eventAttrib);
        void push(const char * message) {
            // do this once
            //static apex_nvtx_range_push_t apex_nvtx_range_push =
                //(apex_nvtx_range_push_t)get_symbol(module, "nvtxRangePushA");
            static apex_nvtx_range_push_ex_t apex_nvtx_range_push =
                (apex_nvtx_range_push_ex_t)get_symbol(module, "nvtxRangePushEx");
            // shouldn't be necessary,
            // but the assertion doesn't happen with release builds
            if (apex_nvtx_range_push != nullptr) {
                //apex_nvtx_range_push(message);
                auto color_id = getColor(message);
                nvtxEventAttributes_t eventAttrib;
                memset(&eventAttrib, 0, sizeof(nvtxEventAttributes_t));
                eventAttrib.version = NVTX_VERSION;
                eventAttrib.size = (uint16_t)sizeof(nvtxEventAttributes_t);
                eventAttrib.colorType = NVTX_COLOR_ARGB;
                eventAttrib.color = colors[color_id];
                eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
                eventAttrib.message.ascii = message;
                apex_nvtx_range_push(&eventAttrib);
            }
        }
        void pop(void) {
            static apex_nvtx_range_pop_t apex_nvtx_range_pop =
                (apex_nvtx_range_pop_t)get_symbol(module, "nvtxRangePop");
            // shouldn't be necessary,
            // but the assertion doesn't happen with release builds
            if (apex_nvtx_range_pop != nullptr) {
                apex_nvtx_range_pop();
            }
        }
    } // namespace apex::dynamic::nvtx

    namespace nvml {
#ifdef __APPLE__
        const char * module ="libapex_cuda.dylib";
#else
        const char * module ="libapex_cuda.so";
#endif
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
                    (apex_nvml_monitor_query_t)get_symbol(module,
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
                    (apex_nvml_monitor_stop_t)get_symbol(module,
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
#ifdef __APPLE__
        const char * module ="libapex_hip.dylib";
#else
        const char * module ="libapex_hip.so";
#endif
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
                    (apex_rsmi_monitor_query_t)get_symbol(module,
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
                    (apex_rsmi_monitor_stop_t)get_symbol(module,
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
                    (apex_rsmi_monitor_getAvailableMemory_t)get_symbol(module,
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
#ifdef __APPLE__
        const char * module ="libapex_hip.dylib";
#else
        const char * module ="libapex_hip.so";
#endif
        void apex_rocprofiler_monitor_query(void);
        void apex_rocprofiler_monitor_stop(void);
        typedef void (*apex_rocprofiler_monitor_query_t)(void);
        typedef void (*apex_rocprofiler_monitor_stop_t)(void);
        void query(void) {
#ifdef APEX_WITH_ROCPROFILER
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rocprofiler_monitor_query_t apex_rocprofiler_monitor_query =
                    (apex_rocprofiler_monitor_query_t)get_symbol(module,
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
#ifdef APEX_WITH_ROCPROFILER
            if (apex_options::monitor_gpu()) {
                // do this once
                static apex_rocprofiler_monitor_stop_t apex_rocprofiler_monitor_stop =
                    (apex_rocprofiler_monitor_stop_t)get_symbol(module,
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
#ifdef __APPLE__
        const char * module ="libapex_hip.dylib";
#else
        const char * module ="libapex_hip.so";
#endif
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
                    (apex_init_hip_tracing_t)get_symbol(module,
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
                    (apex_flush_hip_tracing_t)get_symbol(module,
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
                    (apex_stop_hip_tracing_t)get_symbol(module,
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
#ifdef __APPLE__
        const char * module ="libapex_level0.dylib";
#else
        const char * module ="libapex_level0.so";
#endif
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
                        (apex_init_level0_tracing_t)get_symbol(module,
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
                        (apex_flush_level0_tracing_t)get_symbol(module,
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
                        (apex_stop_level0_tracing_t)get_symbol(module,
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


