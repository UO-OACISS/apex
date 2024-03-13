/* Why does this collection of namespaces (not classes) exist?
 * Well, it exists because we want to have ONE configuration of APEX
 * on a given system, and by not having explicit dependencies on OpenMP,
 * HIP, CUDA, Level0, etc. we can load them dynamically at runtime and
 * provide the support the user is requesting.
 */

#pragma once

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
namespace apex { namespace dynamic {

void * get_symbol(const char * module, const char * symbol);

namespace ompt {
    void do_shutdown(void);
};  // namespace apex::dynamic::ompt

namespace cuda {
    void init(void);
    void flush(void);
    void stop(void);
};  // namespace apex::dynamic::cuda

namespace nvml {
    void query(void);
    void stop(void);
};  // namespace apex::dynamic::rsmi

namespace nvtx {
    void push(const char* message);
    void pop(void);
};  // namespace apex::dynamic::nvtx

namespace roctracer {
    void init(void);
    void flush(void);
    void stop(void);
};  // namespace apex::dynamic::roctracer

namespace rsmi {
    void query(void);
    void stop(void);
    double getAvailableMemory(void);
};  // namespace apex::dynamic::rsmi

namespace rocprofiler {
    void query(void);
    void stop(void);
};  // namespace apex::dynamic::rocprof

namespace level0 {
    void init(void);
    void flush(void);
    void stop(void);
};  // namespace apex::dynamic::level0

}; }; // namespace apex::dynamic


