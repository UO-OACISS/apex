/* https://github.com/kokkos/kokkos-tools/wiki/Profiling-Hooks
 * This page documents the interface between Kokkos and the profiling library.
 * Every function prototype on this page is an interface hook. Profiling
 * libraries may define any subset of the hooks listed here; hooks which are
 * not defined by the library will be silently ignored by Kokkos. The hooks
 * have C linkage, and we emphasize this with the extern "C" required to define
 * such symbols in C++. If the profiling library is written in C, the
 * extern "C" should be omitted.
 */

#include "apex_kokkos.hpp"
#include "apex_api.hpp"
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <stack>
#include <vector>

/*
static std::mutex memory_mtx;
static std::unordered_map<void*,apex::profiler*>& memory_map() {
    static std::unordered_map<void*,apex::profiler*> themap;
    return themap;
}
*/
static std::stack<apex::profiler*>& timer_stack() {
    static APEX_NATIVE_TLS std::stack<apex::profiler*> thestack;
    return thestack;
}
static std::mutex section_mtx;
static std::vector<std::string>& sections() {
    static std::vector<std::string> thevector;
    return thevector;
}
static std::unordered_map<uint32_t, apex::profiler*>& active_sections() {
    static std::unordered_map<uint32_t, apex::profiler*> themap;
    return themap;
}

/* This function will be called only once, prior to calling any other hooks
 * in the profiling library. Currently the only argument which is non-zero
 * is version, which will specify the version of the interface (which will
 * allow future changes to the interface). The version is an integer encoding
 * a date as ((year*100)+month)*100, and the current interface version is
 * 20150628.
 */
extern "C" void kokkosp_init_library(int loadseq, uint64_t version,
    uint32_t ndevinfos, KokkosPDeviceInfo* devinfos) {
    APEX_UNUSED(loadseq);
    APEX_UNUSED(version);
    APEX_UNUSED(ndevinfos);
    APEX_UNUSED(devinfos);
    apex::init("APEX Kokkos handler", 0, 1);
}

/* This function will be called only once, after all other calls to
 * profiling hooks.
 */
extern "C" void kokkosp_finalize_library() {
    apex::finalize();
}

/* These functions are called before their respective parallel constructs
 * execute (Kokkos::parallel_for, Kokkos::parallel_reduce,
 * Kokkos::parallel_scan). The name argument is the name given by the user
 * to the parallel construct, or in the case no name was given it is the
 * compiler-dependent type name of the functor or lambda given to the construct.
 * Currently devid is always zero. kernid is an output variable: the profiling
 * library assigns a value to this, and that value will be given to the
 * corresponding kokkosp_end_parallel_* call at the end of the parallel
 * construct.
 */
extern "C" void kokkosp_begin_parallel_for(const char* name,
    uint32_t devid, uint64_t* kernid) {
    std::stringstream ss;
    ss << "Kokkos for, Dev: " << devid << ", " << name;
    std::string tmp{ss.str()};
    // Start a new profiler, with no known parent
    // (current timer on stack, if exists)
    auto p = apex::start(tmp);
    // save the task wrapper in the kernid
    *(kernid) = (uint64_t)p;
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name,
    uint32_t devid, uint64_t* kernid) {
    std::stringstream ss;
    ss << "Kokkos reduce, Dev: " << devid << ", " << name;
    std::string tmp{ss.str()};
    // Start a new profiler, with no known parent
    // (current timer on stack, if exists)
    auto p = apex::start(tmp);
    // save the task wrapper in the kernid
    *(kernid) = (uint64_t)p;
}

extern "C" void kokkosp_begin_parallel_scan(const char* name,
    uint32_t devid, uint64_t* kernid) {
    std::stringstream ss;
    ss << "Kokkos scan, Dev: " << devid << ", " << name;
    std::string tmp{ss.str()};
    // Start a new profiler, with no known parent
    // (current timer on stack, if exists)
    auto p = apex::start(tmp);
    // save the task wrapper in the kernid
    *(kernid) = (uint64_t)p;
}

extern "C" void kokkosp_end_parallel_for(uint64_t kernid) {
    apex::profiler * p = (apex::profiler*)(kernid);
    apex::stop(p);
}

extern "C" void kokkosp_end_parallel_reduce(uint64_t kernid) {
    apex::profiler * p = (apex::profiler*)(kernid);
    apex::stop(p);
}

extern "C" void kokkosp_end_parallel_scan(uint64_t kernid) {
    apex::profiler * p = (apex::profiler*)(kernid);
    apex::stop(p);
}

/* This function will be called by
 * Kokkos::Profiling::pushRegion(const std::string& cpp_name). The name passed
 * to it is the C equivalent of the cpp_name given to
 * Kokkos::Profiling::pushRegion. As the function names imply, regions are meant
 * to be treated in a stack fashion, ideally consistent with the calling stack
 * of the application. One natural way to use them is to call pushRegion at the
 * beginning of an application function, and call popRegion at the end of the
 * application function. This helps the profiling library group other events
 * like parallel_for calls and memory allocations, and organize them according
 * to the higher-level flow of the application for better presentation to the
 * user.
 */
extern "C" void kokkosp_push_profile_region(const char* name) {
    std::stringstream ss;
    ss << "Kokkos region, " << name;
    std::string tmp{ss.str()};
    // Start a new profiler, with no known parent
    // (current timer on stack, if exists)
    auto p = apex::start(tmp);
    timer_stack().push(p);
}

/* This function will be called by Kokkos::Profiling::popRegion(). In
 * accordance with the stack convention, the region being popped is the one
 * named by the last call to pushRegion.
 */
extern "C" void kokkosp_pop_profile_region() {
    apex::profiler * p = timer_stack().top();
    apex::stop(p);
    timer_stack().pop();
}

/* This function will be called whenever a shared allocation is created to
 * support a Kokkos::View. The handle refers to the Kokkos MemorySpace where
 * the memory resides, the name is the name given by the user to the View. The
 * ptr and size parameters describe the block of memory as its starting pointer
 * and size in bytes.
 */
extern "C" void kokkosp_allocate_data(SpaceHandle_t handle, const char* name,
    void* ptr, uint64_t size) {
    APEX_UNUSED(ptr);
    std::stringstream ss;
    ss << "Kokkos " << handle.name << " data, " << name;
    /*
    std::string tmp{ss.str()};
    auto p = apex::start(tmp);
    memory_mtx.lock();
    memory_map().insert(std::pair<void*,apex::profiler*>(ptr, p));
    memory_mtx.unlock();
    */
    ss << ": Bytes";
    std::string tmp2{ss.str()};
    double bytes = (double)(size);
    apex::sample_value(tmp2, bytes);
}

/* This function will be called whenever a shared allocation is destroyed. The
 * handle refers to the Kokkos MemorySpace where the memory resides, the name is
 * the name given by the user to the View. The ptr and size parameters describe
 * the block of memory as its starting pointer and size in bytes.
 */
extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
    void* ptr, uint64_t size) {
    APEX_UNUSED(handle);
    APEX_UNUSED(name);
    APEX_UNUSED(ptr);
    APEX_UNUSED(size);
    /*
    memory_mtx.lock();
    auto p = memory_map()[ptr];
    memory_map().erase(ptr);
    memory_mtx.unlock();
    apex::stop(p);
    */
}

/* This function will be called whenever a Kokkos::deep_copy function is
 * called on a contiguous view (i.e. it is not a remapping operation from for
 * example one layout to another). The dst_handle and src_handle refer to the
 * MemorySpace of the destination and source allocations respectively. dst_name
 * and src_name are the user provided names of the allocations, while dst_ptr
 * and src_ptr are the respective data pointers. size is the size in bytes of
 * the allocations.
 */
extern "C" void kokkosp_begin_deep_copy(
    SpaceHandle dst_handle, const char* dst_name, const void* dst_ptr,
    SpaceHandle src_handle, const char* src_name, const void* src_ptr,
    uint64_t size) {
    std::stringstream ss;
    ss << "Kokkos deep copy: " << src_handle.name << " " << src_name
       << " -> " << dst_handle.name << " " << dst_name;
    std::string tmp{ss.str()};
    auto p = apex::start(tmp);
    timer_stack().push(p);
    ss << ": Bytes";
    std::string tmp2{ss.str()};
    double bytes = (double)(size);
    apex::sample_value(tmp2, bytes);
    APEX_UNUSED(src_ptr);
    APEX_UNUSED(dst_ptr);
}

/* This function marks the end of a Kokkos::deep_copy call following a
 * kokkosp_begind_deep_copy call.
 */
extern "C" void kokkosp_end_deep_copy() {
    auto p = timer_stack().top();
    apex::stop(p);
    timer_stack().pop();
}

/* Create a profiling section handle. Sections can overlap with each other
 * in contrast to Regions which behave like a stack. name is a user provided
 * name for the section and sec_id is used to return a section identifier to
 * the user. Note that sec_id should be unique for an open section, but name
 * may not be unique. Hence there may be multiple sections with the same name.
 */
extern "C" void kokkosp_create_profile_section( const char* name,
    uint32_t* sec_id) {
    std::string tmp{name};
    section_mtx.lock();
    *sec_id = (uint32_t)(sections().size());
    sections().push_back(tmp);
    section_mtx.unlock();
}

/* Start a profiling section using a previously created section id. A
 * profiling section can be started multiple times, assuming it was first
 * stopped each time.
 */
extern "C" void kokkosp_start_profile_section( uint32_t sec_id) {
    section_mtx.lock();
    auto name = sections()[sec_id];
    section_mtx.unlock();
    auto p = apex::start(name);
    section_mtx.lock();
    active_sections().insert(std::pair<uint32_t, apex::profiler*>(sec_id, p));
    section_mtx.unlock();
}

/* Stop a profiling section using a previously created section id.
 */
extern "C" void kokkosp_stop_profile_section( uint32_t sec_id) {
    section_mtx.lock();
    auto p = active_sections()[sec_id];
    section_mtx.unlock();
    apex::stop(p);
}

/* Destroy a previously created profiling section.
 */
extern "C" void kokkosp_destroy_profile_section( uint32_t sec_id) {
    section_mtx.lock();
    active_sections().erase(sec_id);
    section_mtx.unlock();
}

/* Marks an event during an application with a user provided name.
 */
extern "C" void kokkosp_profile_event( const char* name ) {
    apex::sample_value(name, 0.0);
}

