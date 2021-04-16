/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

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
#include <set>
#include <unordered_map>
#include "apex.hpp"

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

extern "C" {

/* This function will be called only once, prior to calling any other hooks
 * in the profiling library. Currently the only argument which is non-zero
 * is version, which will specify the version of the interface (which will
 * allow future changes to the interface). The version is an integer encoding
 * a date as ((year*100)+month)*100, and the current interface version is
 * 20150628.
 */
void kokkosp_init_library(int loadseq, uint64_t version,
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
void kokkosp_finalize_library() {
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
void kokkosp_begin_parallel_for(const char* name,
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

void kokkosp_begin_parallel_reduce(const char* name,
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

void kokkosp_begin_parallel_scan(const char* name,
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

void kokkosp_end_parallel_for(uint64_t kernid) {
    apex::profiler * p = (apex::profiler*)(kernid);
    apex::stop(p);
}

void kokkosp_end_parallel_reduce(uint64_t kernid) {
    apex::profiler * p = (apex::profiler*)(kernid);
    apex::stop(p);
}

void kokkosp_end_parallel_scan(uint64_t kernid) {
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
void kokkosp_push_profile_region(const char* name) {
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
void kokkosp_pop_profile_region() {
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
void kokkosp_allocate_data(SpaceHandle_t handle, const char* name,
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
void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
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
void kokkosp_begin_deep_copy(
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
void kokkosp_end_deep_copy() {
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
void kokkosp_create_profile_section( const char* name,
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
void kokkosp_start_profile_section( uint32_t sec_id) {
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
void kokkosp_stop_profile_section( uint32_t sec_id) {
    section_mtx.lock();
    auto p = active_sections()[sec_id];
    section_mtx.unlock();
    apex::stop(p);
}

/* Destroy a previously created profiling section.
 */
void kokkosp_destroy_profile_section( uint32_t sec_id) {
    section_mtx.lock();
    active_sections().erase(sec_id);
    section_mtx.unlock();
}

/* Marks an event during an application with a user provided name.
 */
void kokkosp_profile_event( const char* name ) {
    apex::sample_value(name, 0.0);
}

} // extern "C"

#ifdef APEX_HAVE_KOKKOS
//#pragma message("Enabling Kokkos auto-tuning support")

#include "impl/Kokkos_Profiling_C_Interface.h"
#include "apex_api.hpp"
#include "apex_policies.hpp"

std::string pVT(Kokkos_Tools_VariableInfo_ValueType t) {
    if (t == kokkos_value_double) {
        return std::string("double");
    }
    if (t == kokkos_value_int64) {
        return std::string("int64");
    }
    if (t == kokkos_value_string) {
        return std::string("string");
    }
    return std::string("unknown type");
}

std::string pCat(Kokkos_Tools_VariableInfo_StatisticalCategory c) {
    if (c == kokkos_value_categorical) {
        return std::string("categorical");
    }
    if (c == kokkos_value_ordinal) {
        return std::string("ordinal");
    }
    if (c == kokkos_value_interval) {
        return std::string("interval");
    }
    if (c == kokkos_value_ratio) {
        return std::string("ratio");
    }
    return std::string("unknown category");
}

std::string pCVT(Kokkos_Tools_VariableInfo_CandidateValueType t) {
    if (t == kokkos_value_set) {
        return std::string("set");
    }
    if (t == kokkos_value_range) {
        return std::string("range");
    }
    if (t == kokkos_value_unbounded) {
        return std::string("unbounded");
    }
    return std::string("unknown candidate type");
}

std::string pCan(Kokkos_Tools_VariableInfo& i) {
    std::stringstream ss;
    if (i.valueQuantity == kokkos_value_set) {
        std::string delimiter{"["};
        for (size_t index = 0 ; index < i.candidates.set.size ; index++) {
            ss << delimiter;
            if (i.type == kokkos_value_double) {
                ss << i.candidates.set.values.double_value[index];
            } else if (i.type == kokkos_value_int64) {
                ss << i.candidates.set.values.int_value[index];
            } else if (i.type == kokkos_value_string) {
                ss << i.candidates.set.values.string_value[index];
            }
            delimiter = ",";
        }
        ss << "]" << std::endl;
        std::string tmp{ss.str()};
        return tmp;
    }
    if (i.valueQuantity == kokkos_value_range) {
        ss << std::endl;
        if (i.type == kokkos_value_double) {
            ss << "    lower: " << i.candidates.range.lower.double_value << std::endl;
            ss << "    upper: " << i.candidates.range.upper.double_value << std::endl;
            ss << "    step: " << i.candidates.range.step.double_value << std::endl;
        } else if (i.type == kokkos_value_int64) {
            ss << "    lower: " << i.candidates.range.lower.int_value << std::endl;
            ss << "    upper: " << i.candidates.range.upper.int_value << std::endl;
            ss << "    step: " << i.candidates.range.step.int_value << std::endl;
        }
        ss << "    open upper: " << i.candidates.range.openUpper << std::endl;
        ss << "    open lower: " << i.candidates.range.openLower << std::endl;
        std::string tmp{ss.str()};
        return tmp;
    }
    if (i.valueQuantity == kokkos_value_unbounded) {
        return std::string("unbounded\n");
    }
    return std::string("unknown candidate values\n");
}

class Variable {
public:
    Variable(size_t _id, std::string _name, Kokkos_Tools_VariableInfo& _info);
    std::string toString() {
        std::stringstream ss;
        ss << "  name: " << name << std::endl;
        ss << "  id: " << id << std::endl;
        ss << "  info.type: " << pVT(info.type) << std::endl;
        ss << "  info.category: " << pCat(info.category) << std::endl;
        ss << "  info.valueQuantity: " << pCVT(info.valueQuantity) << std::endl;
        ss << "  info.candidates: " << pCan(info) << std::endl;
        std::string tmp{ss.str()};
        return tmp;
    }
    size_t id;
    std::string name;
    Kokkos_Tools_VariableInfo info;
    std::list<double> dspace; // double space
    std::list<uint64_t> lspace; // long space
    std::list<std::string> espace; // enum space
    double dmin;
    double dmax;
    double dstep;
    uint64_t lmin;
    uint64_t lmax;
    uint64_t lstep;
    void makeSpace(void);
};

class KokkosSession {
public:
// EXHAUSTIVE, RANDOM, NELDER_MEAD, PARALLEL_RANK_ORDER
    KokkosSession() :
        window(3),
        strategy(apex_ah_tuning_strategy::NELDER_MEAD),
        verbose(false),
        use_history(false),
        running(false),
        history_file("") {
    }
    int window;
    apex_ah_tuning_strategy strategy;
    std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>>
        requests;
    bool verbose;
    bool use_history;
    bool running;
    std::string history_file;
    std::unordered_map<size_t, Variable*> inputs;
    std::unordered_map<size_t, Variable*> outputs;
    apex_policy_handle * start_policy_handle;
    apex_policy_handle * stop_policy_handle;
    std::unordered_map<size_t, std::string> active_requests;
    std::unordered_map<size_t, uint64_t> context_starts;
};

KokkosSession& getSession() {
    static KokkosSession session;
    return session;
}

Variable::Variable(size_t _id, std::string _name,
    Kokkos_Tools_VariableInfo& _info) : id(_id), name(_name), info(_info) {
    if (getSession().verbose) {
        std::cout << toString();
    }
}

void Variable::makeSpace(void) {
    switch(info.category) {
        case kokkos_value_categorical:
        {
            if (info.valueQuantity == kokkos_value_set) {
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        dspace.push_back(
                                info.candidates.set.values.double_value[index]);
                    } else if (info.type == kokkos_value_int64) {
                        lspace.push_back(
                                info.candidates.set.values.int_value[index]);
                    } else if (info.type == kokkos_value_string) {
                        espace.push_back(
                            std::string(
                                info.candidates.set.values.string_value[index]));
                    }
                }
            }
            break;
        }
        case kokkos_value_interval:
        {
            if (info.valueQuantity == kokkos_value_range) {
                if (info.type == kokkos_value_double) {
                    dstep = info.candidates.range.step.double_value;
                    dmin = info.candidates.range.lower.double_value;
                    dmax = info.candidates.range.upper.double_value;
                    if (info.candidates.range.openLower) {
                        dmin = dmin + dstep;
                    }
                    if (info.candidates.range.openUpper) {
                        dmax = dmax - dstep;
                    }
                } else if (info.type == kokkos_value_int64) {
                    lstep = info.candidates.range.step.int_value;
                    lmin = info.candidates.range.lower.int_value;
                    lmax = info.candidates.range.upper.int_value;
                    if (info.candidates.range.openLower) {
                        lmin = lmin + lstep;
                    }
                    if (info.candidates.range.openUpper) {
                        lmax = lmax - lstep;
                    }
                }
            }
            break;
        }
        case kokkos_value_ordinal:
        case kokkos_value_ratio:
        default:
        {
            break;
        }
    }
}

int register_policy() {
    return APEX_NOERROR;
}

extern "C" {
/*
 * In the past, tools have responded to the profiling hooks in Kokkos.
 * This effort adds to that, there are now a few more functions (note
 * that I'm using the C names for types. In general you can replace
 * Kokkos_Tools_ with Kokkos::Tools:: in C++ tools)
 *
 */

size_t& getDepth() {
    static size_t depth{0};
    return depth;
}

/* Declares a tuning variable named name with uniqueId id and all the
 * semantic information stored in info. Note that the VariableInfo
 * struct has a void* field called toolProvidedInfo. If you fill this
 * in, every time you get a value of that type you'll also get back
 * that same pointer.
 */
void kokkosp_declare_output_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info) {
    std::cout << std::string(getDepth(), ' ');
    std::cout << __func__ << std::endl;
    Variable * output = new Variable(id, name, info);
    output->makeSpace();
    getSession().outputs.insert(std::make_pair(id, output));
    return;
}

/* This is almost exactly like declaring a tuning variable. The only
 * difference is that in cases where the candidate values aren't known,
 * info.valueQuantity will be set to kokkos_value_unbounded. This is
 * fairly common, Kokkos can tell you that kernel_name is a string,
 * but we can't tell you what strings a user might provide.
 */
void kokkosp_declare_input_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info) {
    std::cout << std::string(getDepth(), ' ');
    std::cout << __func__ << std::endl;
    Variable * input = new Variable(id, name, info);
    getSession().inputs.insert(std::make_pair(id, input));
}

void printContext(size_t numVars, const Kokkos_Tools_VariableValue* values) {
    std::cout << ", cv: " << numVars;
    std::string d{"["};
    for (size_t i = 0 ; i < numVars ; i++) {
        auto id = values[i].type_id;
        std::cout << d << id << ":";
        Variable* var{getSession().inputs[id]};
        switch (var->info.type) {
            case kokkos_value_double:
                std::cout << values[i].value.double_value;
                break;
            case kokkos_value_int64:
                std::cout << values[i].value.int_value;
                break;
            case kokkos_value_string:
                std::cout << values[i].value.string_value;
                break;
            default:
                break;
        }
        d = ",";
    }
    std::cout << "]";
}

std::string hashContext(size_t numVars, const Kokkos_Tools_VariableValue* values) {
    std::stringstream ss;
    std::string d{"["};
    for (size_t i = 0 ; i < numVars ; i++) {
        auto id = values[i].type_id;
        ss << d << id << ":";
        Variable* var{getSession().inputs[id]};
        switch (var->info.type) {
            case kokkos_value_double:
                ss << values[i].value.double_value;
                break;
            case kokkos_value_int64:
                ss << values[i].value.int_value;
                break;
            case kokkos_value_string:
                ss << values[i].value.string_value;
                break;
            default:
                break;
        }
        d = ",";
    }
    ss << "]";
    std::string tmp{ss.str()};
    return tmp;
}

void printTuning(const size_t numVars, Kokkos_Tools_VariableValue* values) {
    std::cout << "tv: " << numVars;
    std::string d{"["};
    for (size_t i = 0 ; i < numVars ; i++) {
        auto id = values[i].type_id;
        std::cout << d << id << ":";
        Variable* var{getSession().outputs[id]};
        switch (var->info.type) {
            case kokkos_value_double:
                std::cout << values[i].value.double_value;
                break;
            case kokkos_value_int64:
                std::cout << values[i].value.int_value;
                break;
            case kokkos_value_string:
                std::cout << values[i].value.string_value;
                break;
            default:
                break;
        }
        d = ",";
    }
    std::cout << "]";
    std::cout << std::endl;
}

void set_params(std::shared_ptr<apex_tuning_request> request,
    const size_t vars,
    Kokkos_Tools_VariableValue* values) {
    APEX_UNUSED(request);
    for (size_t i = 0 ; i < vars ; i++) {
        auto id = values[i].type_id;
        Variable* var{getSession().outputs[id]};
        if (var->info.type == kokkos_value_double) {
            auto thread_param = std::static_pointer_cast<apex_param_double>(
                request->get_param(var->name));
            values[i].value.double_value = thread_param->get_value();
        } else if (var->info.type == kokkos_value_int64) {
            auto thread_param = std::static_pointer_cast<apex_param_long>(
                request->get_param(var->name));
            values[i].value.int_value = thread_param->get_value();
        } else if (var->info.type == kokkos_value_string) {
            auto thread_param = std::static_pointer_cast<apex_param_enum>(
                request->get_param(var->name));
            strncpy(values[i].value.string_value, thread_param->get_value().c_str(), 64);
        }
    }
}

void handle_start(const std::string & name, const size_t vars,
    Kokkos_Tools_VariableValue* values) {
    KokkosSession& session = getSession();
    auto search = session.requests.find(name);
    if(search == session.requests.end()) {
        // Start a new tuning session.
        if(session.verbose) {
            fprintf(stderr, "Starting tuning session for %s\n", name.c_str());
        }
        std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
        session.requests.insert(std::make_pair(name, request));

        // Create an event to trigger this tuning session.
        apex_event_type trigger = apex::register_custom_event(name);
        request->set_trigger(trigger);

        // Create a metric
        std::function<double(void)> metric = [=]()->double{
            apex_profile * profile = apex::get_profile(name);
            if(profile == nullptr) {
                std::cerr << "ERROR: no profile for " << name << std::endl;
                return 0.0;
            }
            if(profile->calls == 0.0) {
                std::cerr << "ERROR: calls = 0 for " << name << std::endl;
                return 0.0;
            }
            double result = profile->accumulated/profile->calls;
            //if(session.verbose) {
                fprintf(stderr, "time per call: %f\n", (double)(result)/1000000000.0);
            //}
            return result;
        };
        request->set_metric(metric);

        // Set apex_openmp_policy_tuning_strategy
        request->set_strategy(session.strategy);

        for (size_t i = 0 ; i < vars ; i++) {
            auto id = values[i].type_id;
            Variable* var{getSession().outputs[id]};
            if (var->info.type == kokkos_value_double) {
                std::cout << session.outputs[id]->name << " init: " <<
                    session.outputs[id]->dmin << " min: " <<
                    session.outputs[id]->dmin << " max: " <<
                    session.outputs[id]->dmax << " step: " <<
                    session.outputs[id]->dstep << std::endl;
                auto tmp = request->add_param_double(
                    session.outputs[id]->name,
                    session.outputs[id]->dmin,
                    session.outputs[id]->dmin,
                    session.outputs[id]->dmax,
                    session.outputs[id]->dstep);
            } else if (var->info.type == kokkos_value_int64) {
                auto tmp = request->add_param_long(
                    session.outputs[id]->name,
                    session.outputs[id]->lmin,
                    session.outputs[id]->lmin,
                    session.outputs[id]->lmax,
                    session.outputs[id]->lstep);
            } else if (var->info.type == kokkos_value_string) {
                auto tmp = request->add_param_enum(
                    session.outputs[id]->name,
                    session.outputs[id]->espace.front(),
                    session.outputs[id]->espace);
            }
        }

        // Set OpenMP runtime parameters to initial values.
        set_params(request, vars, values);

        // Start the tuning session.
        apex::setup_custom_tuning(*request);
    } else {
        // We've seen this region before.
        std::shared_ptr<apex_tuning_request> request = search->second;
        set_params(request, vars, values);
    }
}

void handle_stop(const std::string & name) {
    auto search = getSession().requests.find(name);
    if(search == getSession().requests.end()) {
        std::cerr << "ERROR: No data for \"" << name << std::endl;
    } else {
        apex_profile * profile = apex::get_profile(name);
        if(getSession().window == 1 ||
           (profile != nullptr &&
            profile->calls >= getSession().window)) {
            //std::cout << "Num calls: " << profile->calls << std::endl;
            std::shared_ptr<apex_tuning_request> request = search->second;
            // Evaluate the results
            apex::custom_event(request->get_trigger(), NULL);
            // Reset counter so each measurement is fresh.
            apex::reset(name);
        }
    }
}

/* Here Kokkos is requesting the values of tuning variables, and most
 * of the meat is here. The contextId tells us the scope across which
 * these variables were used.
 *
 * The next two arguments describe the context you're tuning in. You
 * have the number of context variables, and an array of that size
 * containing their values. Note that the Kokkos_Tuning_VariableValue
 * has a field called metadata containing all the info (type,
 * semantics, and critically, candidates) about that variable.
 *
 * The two arguments following those describe the Tuning Variables.
 * First the number of them, then an array of that size which you can
 * overwrite. Overwriting those values is how you give values back to
 * the application.
 *
 * Critically, as tuningVariableValues comes preloaded with default
 * values, if your function body is return; you will not crash Kokkos,
 * only make us use our defaults. If you don't know, you are allowed
 * to punt and let Kokkos do what it would.
 */
void kokkosp_request_values(
    const size_t contextId,
    const size_t numContextVariables,
    const Kokkos_Tools_VariableValue* contextVariableValues,
    const size_t numTuningVariables,
    Kokkos_Tools_VariableValue* tuningVariableValues) {
    if (getSession().verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __func__ << " ctx: " << contextId;
        printContext(numContextVariables, contextVariableValues);
    }
    std::string name{hashContext(numContextVariables, contextVariableValues)};
    handle_start(name, numTuningVariables, tuningVariableValues);
    getSession().active_requests.insert(std::pair<uint32_t, std::string>(contextId, name));
    //if (getSession().verbose) {
        //std::cout << std::endl << std::string(getDepth(), ' ');
        printTuning(numTuningVariables, tuningVariableValues);
    //}
    // throw away the time spent in this step!
    getSession().context_starts[contextId] = apex::profiler::now_ns();
}

/* This starts the context pointed at by contextId. If tools use
 * measurements to drive tuning, this is where they'll do their
 * starting measurement.
 */
void kokkosp_begin_context(size_t contextId) {
    if (getSession().verbose) {
        std::cout << std::string(getDepth()++, ' ');
        std::cout << __func__ << "\t" << contextId << std::endl;
    }
    std::stringstream ss;
    getSession().context_starts.insert(
        std::pair<uint32_t, uint64_t>(contextId, apex::profiler::now_ns()));
}

/* This simply says that the contextId in the argument is now over.
 * If you provided tuning values associated with that context, those
 * values can now be associated with a result.
 */
void kokkosp_end_context(const size_t contextId) {
    if (getSession().verbose) {
        std::cout << std::string(--getDepth(), ' ');
        std::cout << __func__ << "\t" << contextId << std::endl;
    }
    uint64_t end = apex::profiler::now_ns();
    auto start = getSession().context_starts.find(contextId);
    auto name = getSession().active_requests.find(contextId);
    if (name != getSession().active_requests.end() &&
        start != getSession().context_starts.end()) {
        apex::sample_value(name->second, (double)(end-start->second));
        handle_stop(name->second);
        getSession().active_requests.erase(contextId);
        getSession().context_starts.erase(contextId);
    }
}

#endif // APEX_HAVE_KOKKOS

} // extern "C"
