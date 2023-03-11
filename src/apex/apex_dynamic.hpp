#pragma once

/* OMPT will be automatically initialized by the OpenMP runtime,
 * so we don't need to dynamically connect to a startup function.
 * However, we do need to connect to a finalize function. */
namespace apex { namespace dynamic {

namespace ompt {
    void do_shutdown(void);
};  // namespace apex::dynamic::ompt

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

namespace rocprof {
    void query(void);
    void stop(void);
};  // namespace apex::dynamic::rocprof

}; }; // namespace apex::dynamic


