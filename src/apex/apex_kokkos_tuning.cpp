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
#include <stdlib.h>
#include "apex.hpp"
#include "Kokkos_Profiling_C_Interface.h"
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
    std::list<std::string> space; // enum space
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
        strategy(apex_ah_tuning_strategy::SIMULATED_ANNEALING),
        verbose(false),
        use_history(false),
        running(false),
        history_file("") {
            verbose = apex::apex_options::use_kokkos_verbose();
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
        case kokkos_value_ordinal:
        {
            if (info.valueQuantity == kokkos_value_set) {
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.double_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_int64) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.int_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_string) {
                        space.push_back(
                            std::string(
                                info.candidates.set.values.string_value[index]));
                    }
                }
            }
            break;
        }
        case kokkos_value_interval:
        case kokkos_value_ratio:
        {
            if (info.valueQuantity == kokkos_value_range) {
                if (info.type == kokkos_value_double) {
                    dstep = info.candidates.range.step.double_value;
                    dmin = info.candidates.range.lower.double_value;
                    dmax = info.candidates.range.upper.double_value;
                    if (info.candidates.range.openLower == 0) {
                        dmin = dmin + dstep;
                    }
                    if (info.candidates.range.openUpper == 0) {
                        dmax = dmax - dstep;
                    }
                } else if (info.type == kokkos_value_int64) {
                    lstep = info.candidates.range.step.int_value;
                    lmin = info.candidates.range.lower.int_value;
                    lmax = info.candidates.range.upper.int_value;
                    if (info.candidates.range.openLower == 0) {
                        lmin = lmin + lstep;
                    }
                    if (info.candidates.range.openUpper == 0) {
                        lmax = lmax - lstep;
                    }
                }
            }
            if (info.valueQuantity == kokkos_value_set) {
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.double_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_int64) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.int_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_string) {
                        space.push_back(
                            std::string(
                                info.candidates.set.values.string_value[index]));
                    }
                }
            }
            break;
        }
        default:
        {
            break;
        }
    }
}

int register_policy() {
    return APEX_NOERROR;
}

size_t& getDepth() {
    static size_t depth{0};
    return depth;
}

std::string hashContext(size_t numVars,
    const Kokkos_Tools_VariableValue* values,
    std::unordered_map<size_t, Variable*>& varmap) {
    std::stringstream ss;
    std::string d{"["};
    for (size_t i = 0 ; i < numVars ; i++) {
        auto id = values[i].type_id;
        ss << d << id << ":";
        Variable* var{varmap[id]};
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

void printContext(size_t numVars, const Kokkos_Tools_VariableValue* values) {
    std::cout << ", cv: " << numVars;
    std::cout << hashContext(numVars, values, getSession().inputs);
}

void printTuning(const size_t numVars, Kokkos_Tools_VariableValue* values) {
    std::cout << "tv: " << numVars;
    std::cout << hashContext(numVars, values, getSession().outputs);
    std::cout << std::endl;
}

void set_params(std::shared_ptr<apex_tuning_request> request,
    const size_t vars,
    Kokkos_Tools_VariableValue* values) {
    APEX_UNUSED(request);
    for (size_t i = 0 ; i < vars ; i++) {
        auto id = values[i].type_id;
        Variable* var{getSession().outputs[id]};
        if (var->info.valueQuantity == kokkos_value_set) {
            auto thread_param = std::static_pointer_cast<apex_param_enum>(
                request->get_param(var->name));
            if (var->info.type == kokkos_value_double) {
                values[i].value.double_value = std::stod(thread_param->get_value());
                std::string tmp(request->get_name()+":"+var->name);
                apex::sample_value(tmp, values[i].value.double_value);
            } else if (var->info.type == kokkos_value_int64) {
                values[i].value.int_value = std::stol(thread_param->get_value());
                std::string tmp(request->get_name()+":"+var->name);
                apex::sample_value(tmp, values[i].value.int_value);
            } else if (var->info.type == kokkos_value_string) {
                strncpy(values[i].value.string_value, thread_param->get_value().c_str(), 64);
            }
        } else { // range
            if (var->info.type == kokkos_value_double) {
                auto thread_param = std::static_pointer_cast<apex_param_double>(
                    request->get_param(var->name));
                values[i].value.double_value = thread_param->get_value();
                std::string tmp(request->get_name()+":"+var->name);
                apex::sample_value(tmp, values[i].value.double_value);
            } else if (var->info.type == kokkos_value_int64) {
                auto thread_param = std::static_pointer_cast<apex_param_long>(
                    request->get_param(var->name));
                values[i].value.int_value = thread_param->get_value();
                std::string tmp(request->get_name()+":"+var->name);
                apex::sample_value(tmp, values[i].value.int_value);
            }
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
            if(session.verbose) {
                fprintf(stdout, "time per call: %fs\n", (double)(result)/1000000000.0);
            }
            return result;
        };
        request->set_metric(metric);

        // Set apex_openmp_policy_tuning_strategy
        request->set_strategy(session.strategy);
        request->set_radius(0.5);
        request->set_aggregation_times(3);
        // min, max, mean
        request->set_aggregation_function("min");

        for (size_t i = 0 ; i < vars ; i++) {
            auto id = values[i].type_id;
            Variable* var{getSession().outputs[id]};
            /* If it's a set, the initial value can be a double, int or string
             * because we store all interval sets as enumerations of strings */
            if (var->info.valueQuantity == kokkos_value_set) {
                std::list<std::string>& space = session.outputs[id]->space;
                std::string front;
                if (var->info.type == kokkos_value_double) {
                    front = std::to_string(values[i].value.double_value);
                } else if (var->info.type == kokkos_value_int64) {
                    front = std::to_string(values[i].value.int_value);
                } else if (var->info.type == kokkos_value_int64) {
                    front = std::string(values[i].value.string_value);
                }
                auto tmp = request->add_param_enum(
                    session.outputs[id]->name, front, space);
            } else {
                if (var->info.type == kokkos_value_double) {
                    auto tmp = request->add_param_double(
                        session.outputs[id]->name,
                        values[i].value.double_value,
                        session.outputs[id]->dmin,
                        session.outputs[id]->dmax,
                        session.outputs[id]->dstep);
                } else if (var->info.type == kokkos_value_int64) {
                    auto tmp = request->add_param_long(
                        session.outputs[id]->name,
                        values[i].value.int_value,
                        session.outputs[id]->lmin,
                        session.outputs[id]->lmax,
                        session.outputs[id]->lstep);
                }
            }
        }
        std::cout << std::endl;

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

extern "C" {
/*
 * In the past, tools have responded to the profiling hooks in Kokkos.
 * This effort adds to that, there are now a few more functions (note
 * that I'm using the C names for types. In general you can replace
 * Kokkos_Tools_ with Kokkos::Tools:: in C++ tools)
 *
 */

/* Declares a tuning variable named name with uniqueId id and all the
 * semantic information stored in info. Note that the VariableInfo
 * struct has a void* field called toolProvidedInfo. If you fill this
 * in, every time you get a value of that type you'll also get back
 * that same pointer.
 */
void kokkosp_declare_output_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info) {
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    //if (!apex::apex_options::use_kokkos_tuning()) { return; }
    if(getSession().verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __func__ << std::endl;
    }
    Variable * output = new Variable(id, name, info);
    output->makeSpace();
    getSession().outputs.insert(std::make_pair(id, output));
    getSession().inputs.insert(std::make_pair(id, output));
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
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    //if (!apex::apex_options::use_kokkos_tuning()) { return; }
    if(getSession().verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __func__ << std::endl;
    }
    Variable * input = new Variable(id, name, info);
    getSession().inputs.insert(std::make_pair(id, input));
    getSession().outputs.insert(std::make_pair(id, input));
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
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    if (getSession().verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __func__ << " ctx: " << contextId;
        printContext(numContextVariables, contextVariableValues);
    }
    std::string name{hashContext(numContextVariables, contextVariableValues,
        getSession().inputs)};
    getSession().active_requests.insert(
        std::pair<uint32_t, std::string>(contextId, name));
    if (apex::apex_options::use_kokkos_tuning()) {
        handle_start(name, numTuningVariables, tuningVariableValues);
    }
    if (getSession().verbose) {
        std::cout << std::endl << std::string(getDepth(), ' ');
        printTuning(numTuningVariables, tuningVariableValues);
    }
    // throw away the time spent in this step!
    getSession().context_starts[contextId] = apex::profiler::now_ns();
}

/* This starts the context pointed at by contextId. If tools use
 * measurements to drive tuning, this is where they'll do their
 * starting measurement.
 */
void kokkosp_begin_context(size_t contextId) {
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    //if (!apex::apex_options::use_kokkos_tuning()) { return; }
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
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
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
        if (apex::apex_options::use_kokkos_tuning()) {
            handle_stop(name->second);
        }
        getSession().active_requests.erase(contextId);
        getSession().context_starts.erase(contextId);
    }
}

} // extern "C"

