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
#include <map>
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

class Bin {
public:
    Bin(double value, size_t idx) :
        mean((double)value),
        total(value),
        min(value),
        max(value),
        count(1) {
        std::stringstream ss;
        ss << "bin_" << idx;
        name = ss.str();
    }
    double mean;
    double total;
    double min;
    double max;
    size_t count;
    std::string name;
    bool contains(double value) {
        if (value <= max && value >= min) {
            return true;
        } else if (value <= (mean * 1.25) &&
                   value >= (mean * 0.75)) {
            return true;
        }
        return false;
    }
    void add(double value) {
        count++;
        total += value;
        mean = (double)total / (double)count;
        if (value < min) min = value;
        if (value > max) max = value;
    }
    std::string getName() {
        return name;
    }
};


class Variable {
public:
    Variable(size_t _id, std::string _name, Kokkos_Tools_VariableInfo& _info);
    void deepCopy(Kokkos_Tools_VariableInfo& _info);
    std::string toString() {
        std::stringstream ss;
        ss << "  hash: " << hashValue << std::endl;
        ss << "  name: " << name << std::endl;
        ss << "  id: " << id << std::endl;
        ss << "  info.type: " << pVT(info.type) << std::endl;
        ss << "  info.category: " << pCat(info.category) << std::endl;
        ss << "  info.valueQuantity: " << pCVT(info.valueQuantity) << std::endl;
        ss << "  info.candidates: " << pCan(info);
        if (info.valueQuantity == kokkos_value_unbounded) {
            ss << "  num_bins: " << bins.size() << std::endl;
            for (auto b : bins) {
                ss << "  " << b->name << ": " << std::endl;
                ss << "    min: " << std::fixed << b->min << std::endl;
                ss << "    mean: " << std::fixed << b->mean << std::endl;
                ss << "    max: " << std::fixed << b->max << std::endl;
                ss << "    count: " << std::fixed << b->count << std::endl;
            }
        }
        std::string tmp{ss.str()};
        return tmp;
    }
    size_t id;
    std::string name;
    std::string hashValue;
    Kokkos_Tools_VariableInfo info;
    std::list<std::string> space; // enum space
    double dmin;
    double dmax;
    double dstep;
    int64_t lmin;
    int64_t lmax;
    int64_t lstep;
    int64_t lvar;
    int64_t numValues;
    void makeSpace(void);
    std::vector<Bin*> bins;
    std::string getBin(double value) {
        for (auto b : bins) {
            if (b->contains(value)) {
                b->add(value);
                return b->getName();
            }
        }
        Bin * b = new Bin(value, bins.size());
        std::string tmp{b->getName()};
        bins.push_back(b);
        return tmp;
    }
};

/* this class helps us with nested contexts. We don't want to
 * stop searching a higher level branch until all of the subtrees
 * have also converged - the global optimum might be at the bottom
 * of a branch with many options, but we won't get there if we
 * have found local minimum in a simpler branch. */
class TreeNode {
    static std::map<std::string,TreeNode*> allContexts;
public:
    TreeNode(std::string _name) :
        name(_name), _hasConverged(false) {}
    static TreeNode* find(std::string name, TreeNode* parent) {
        auto node = allContexts.find(name);
        if (node == allContexts.end()) {
            TreeNode *tmp = new TreeNode(name);
            if (parent != nullptr) {
                parent->children.insert(tmp);
            }
            allContexts.insert(std::pair<std::string, TreeNode*>(name,tmp));
            return tmp;
        }
        return node->second;
    }
    std::string name;
    std::set<TreeNode*> children;
    bool _hasConverged;
    bool haveChildren(void) {
        return (children.size() > 0);
    }
    bool childrenConverged(void) {
        // yes children? have they all converged?
        for (auto child : children) {
            if (!child->childrenConverged()) { return false; }
            if (!child->_hasConverged) { return false; }
        }
        return true;
    }
};

std::map<std::string,TreeNode*> TreeNode::allContexts;

class KokkosSession {
private:
// EXHAUSTIVE, RANDOM, NELDER_MEAD, PARALLEL_RANK_ORDER
    KokkosSession() :
        window(apex::apex_options::kokkos_tuning_window()),
        verbose(false),
        use_history(false),
        running(false){
            verbose = apex::apex_options::use_kokkos_verbose();
            // don't do this until the object is constructed!
            saveCache = (apex::apex::instance()->get_node_id() == 0) ? true : false;
            if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "random", strlen("random")) == 0) {
                strategy = apex_ah_tuning_strategy::APEX_RANDOM;
            } else if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "exhaustive", strlen("exhaustive")) == 0) {
                strategy = apex_ah_tuning_strategy::APEX_EXHAUSTIVE;
            } else if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "simulated_annealing", strlen("simulated_annealing")) == 0) {
                strategy = apex_ah_tuning_strategy::SIMULATED_ANNEALING;
            } else if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "genetic_search", strlen("genetic_search")) == 0) {
                strategy = apex_ah_tuning_strategy::GENETIC_SEARCH;
            } else if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "nelder_mead", strlen("nelder_mead")) == 0) {
                strategy = apex_ah_tuning_strategy::NELDER_MEAD_INTERNAL;
            } else if (strncmp(apex::apex_options::kokkos_tuning_policy(),
                    "automatic", strlen("automatic")) == 0) {
                strategy = apex_ah_tuning_strategy::AUTOMATIC;
            } else {
                strategy = apex_ah_tuning_strategy::AUTOMATIC;
            }
            saved_node_id = apex::apex::instance()->get_node_id();
    }
public:
    ~KokkosSession() {
        writeCache();
    }
    static KokkosSession& getSession();
    KokkosSession(const KokkosSession&) =delete;
    KokkosSession& operator=(const KokkosSession&) =delete;
    int window;
    bool verbose;
    bool use_history;
    bool running;
    bool saveCache;
    apex_ah_tuning_strategy strategy;
    std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>>
        requests;
    std::unordered_map<std::string, std::vector<int>> var_ids;
    std::map<size_t, Variable*> inputs;
    std::map<size_t, Variable*> outputs;
    std::map<size_t, Variable*> all_vars;
    apex_policy_handle * start_policy_handle;
    apex_policy_handle * stop_policy_handle;
    std::unordered_map<size_t, std::string> active_requests;
    std::set<size_t> used_history;
    std::unordered_map<size_t, uint64_t> context_starts;
    std::stack<TreeNode*> contextStack;
    void writeCache();
    bool checkForCache();
    void readCache();
    void saveInputVar(size_t id, Variable * var);
    void saveOutputVar(size_t id, Variable * var);
    void parseVariableCache(std::ifstream& results);
    void parseContextCache(std::ifstream& results);
    std::string makeCacheFileName(void);
    //std::stringstream cachedResults;
    std::string cacheFilename;
    // Map from cached variables to variable values
    std::map<std::string, struct Kokkos_Tools_VariableInfo> cachedVariables;
    // map from cached variable IDs to cached variable names
    std::map<size_t, std::string> cachedVariableNames;
    // map from cached variable names to cached variable IDs
    std::map<std::string, size_t> cachedVariableIDs;
    std::map<std::string, std::map<size_t, struct Kokkos_Tools_VariableValue> > cachedTunings;
    int saved_node_id;
};

std::string KokkosSession::makeCacheFileName(void) {
    std::string filename{"./apex_converged_tuning."};
    filename += std::to_string(saved_node_id);
    filename += std::string{".yaml"};
    return filename;
}

/* If we've cached values, we can bypass a lot. */
bool KokkosSession::checkForCache() {
    static bool once{false};
    if (once) { return use_history; }
    once = true;
    // did the user specify a file?
    if (strlen(apex::apex_options::kokkos_tuning_cache()) > 0) {
        cacheFilename = std::string(apex::apex_options::kokkos_tuning_cache());
    } else {
        cacheFilename = makeCacheFileName();
    }
    std::ifstream f(cacheFilename);
    if (f.good()) {
        use_history = true;
        if(verbose) {
            std::cout << "Cache found" << std::endl;
        }
        readCache();
    } else {
        if(verbose) {
            std::cout << "Cache not found" << std::endl;
        }
    }
    return use_history;
}

void KokkosSession::saveInputVar(size_t id, Variable * var) {
    // insert the id given to us
    inputs.insert(std::make_pair(id, var));
    all_vars.insert(std::make_pair(id, var));
}

void KokkosSession::saveOutputVar(size_t id, Variable * var) {
    outputs.insert(std::make_pair(id, var));
    all_vars.insert(std::make_pair(id, var));
}

std::string strategy_to_string(std::shared_ptr<apex_tuning_request> request) {
    if (request->get_strategy() == apex_ah_tuning_strategy::APEX_RANDOM) {
        return std::string("random");
    }
    if (request->get_strategy() == apex_ah_tuning_strategy::APEX_EXHAUSTIVE) {
        return std::string("exhaustive");
    }
    if (request->get_strategy() == apex_ah_tuning_strategy::SIMULATED_ANNEALING) {
        return std::string("simulated annealing");
    }
    if (request->get_strategy() == apex_ah_tuning_strategy::GENETIC_SEARCH) {
        return std::string("genetic search");
    }
    if (request->get_strategy() == apex_ah_tuning_strategy::NELDER_MEAD_INTERNAL) {
        return std::string("nelder mead");
    }
    return "unknown?";
}

void KokkosSession::writeCache(void) {
    if(use_history) { return; }
    //if(!saveCache) { return; }
    // did the user specify a file?
    if (strlen(apex::apex_options::kokkos_tuning_cache()) > 0) {
        cacheFilename = std::string(apex::apex_options::kokkos_tuning_cache());
    } else {
        cacheFilename = makeCacheFileName();
    }
    std::ofstream results(cacheFilename);
    std::cout << "Writing cache of Kokkos tuning results to: '" << cacheFilename << "'" << std::endl;
    for (auto i : inputs) {
        size_t id = i.first;
        Variable* v = i.second;
        results << "Input_" << id << ":" << std::endl;
        results << v->toString();
    }
    for (auto o : outputs) {
        size_t id = o.first;
        Variable* v = o.second;
        results << "Output_" << id << ":" << std::endl;
        results << v->toString();
    }
    size_t count = 0;
    for (const auto &req : requests) {
        results << "Context_" << count++ << ":" << std::endl;
        results << "  Name: \"" << req.first << "\"" << std::endl;
        std::shared_ptr<apex_tuning_request> request = req.second;
        // always write the random search out
        bool converged = request->has_converged() ||
            strategy == apex_ah_tuning_strategy::APEX_RANDOM;
        results << "  Strategy: \"" <<
            strategy_to_string(request);
        if (strategy == apex_ah_tuning_strategy::AUTOMATIC) {
            results << " (auto)";
        }
        results << "\"" << std::endl;
        results << "  Converged: " <<
            (converged ? "true" : "false") << std::endl;
        if (converged) {
            results << "  Results:" << std::endl;
            results << "    NumVars: " << var_ids[req.first].size() << std::endl;
            for (const auto &id : var_ids[req.first]) {
                results << "    id: " << id << std::endl;
                Variable* var{KokkosSession::getSession().outputs[id]};
                if (var->info.valueQuantity == kokkos_value_set) {
                    auto param = std::static_pointer_cast<apex_param_enum>(
                        request->get_param(var->name));
                    results << "    value: " << param->get_value() << std::endl;
                } else if (var->info.valueQuantity == kokkos_value_range) {
                    if (var->info.type == kokkos_value_double) {
                        auto param = std::static_pointer_cast<apex_param_double>(
                            request->get_param(var->name));
                        results << "    value: " << param->get_value() << std::endl;
                    } else if (var->info.type == kokkos_value_int64) {
                        auto param = std::static_pointer_cast<apex_param_long>(
                            request->get_param(var->name));
                        results << "    value: " << param->get_value() << std::endl;
                    }
                }
            }
        }
        // if not converged, need to get the "best so far" values for the parameters.
    }
    results.close();
}

void KokkosSession::parseVariableCache(std::ifstream& results) {
    std::string line;
    std::string delimiter = ": ";
    struct Kokkos_Tools_VariableInfo info;
    memset(&info, 0, sizeof(struct Kokkos_Tools_VariableInfo));
    // name
    std::getline(results, line);
    std::string name = line.substr(line.find(delimiter)+2);
    // id
    std::getline(results, line);
    size_t id = atol(line.substr(line.find(delimiter)+2).c_str());
    // info.type
    std::getline(results, line);
    std::string type = line.substr(line.find(delimiter)+2);
    if (type.find("double") != std::string::npos) {
        info.type = kokkos_value_double;
    } else if (type.find("int64") != std::string::npos) {
        info.type = kokkos_value_int64;
    } else if (type.find("string") != std::string::npos) {
        info.type = kokkos_value_string;
    }
    // info.category
    std::getline(results, line);
    std::string category = line.substr(line.find(delimiter)+2);
    if (category.find("categorical") != std::string::npos) {
        info.category = kokkos_value_categorical;
    } else if (category.find("ordinal") != std::string::npos) {
        info.category = kokkos_value_ordinal;
    } else if (category.find("interval") != std::string::npos) {
        info.category = kokkos_value_interval;
    } else if (category.find("ratio") != std::string::npos) {
        info.category = kokkos_value_ratio;
    }
    // info.valueQuantity
    std::getline(results, line);
    std::string valueQuantity = line.substr(line.find(delimiter)+2);
    if (valueQuantity.find("set") != std::string::npos) {
        info.valueQuantity = kokkos_value_set;
        info.candidates.set.size = 0;
    } else if (valueQuantity.find("range") != std::string::npos) {
        info.valueQuantity = kokkos_value_range;
        info.candidates.set.size = 0;
    } else if (valueQuantity.find("unbounded") != std::string::npos) {
        info.valueQuantity = kokkos_value_unbounded;
    }
    // info.candidates
    std::getline(results, line);
    std::string candidates = line.substr(line.find(delimiter)+2);
    // ...eh, who cares.  We don't need the candidate values.
    // map the name to the variable
    cachedVariables.insert(std::make_pair(name, std::move(info)));
    // map the old id to the name
    cachedVariableNames.insert(std::make_pair(id, name));
    // map the name to the old id
    cachedVariableIDs.insert(std::make_pair(name, id));
    /*
    if (candidates.find("unbounded") != std::string::npos) {
        info.candidates = kokkos_value_unbounded;
        // bin_*: - can be skipped
        std::getline(results, line);
        //min: 3751.000000
        std::getline(results, line);
        std::string min = line.substr(line.find(delimiter)+2);
        //mean: 4094.721536
        std::getline(results, line);
        std::string mean = line.substr(line.find(delimiter)+2);
        //max: 4480.000000
        std::getline(results, line);
        std::string max = line.substr(line.find(delimiter)+2);
        //count: 729
        std::getline(results, line);
        std::string count = line.substr(line.find(delimiter)+2);
    }
    */
}

void KokkosSession::parseContextCache(std::ifstream& results) {
    std::string line;
    std::string delimiter = ": ";
    // name
    std::getline(results, line);
    std::string name = line.substr(line.find(delimiter)+2);
    name.erase(std::remove(name.begin(),name.end(),'\"'),name.end());
    // strategy
    std::getline(results, line);
    // converged?
    std::getline(results, line);
    std::string converged = line.substr(line.find(delimiter)+2);
    if (converged.find("true") != std::string::npos) {
        std::map<size_t, struct Kokkos_Tools_VariableValue> vars;
        // Results
        std::getline(results, line);
        // NumVars
        std::getline(results, line);
        size_t numvars = atol(line.substr(line.find(delimiter)+2).c_str());
        for (size_t i = 0 ; i < numvars ; i++) {
            struct Kokkos_Tools_VariableValue var;
            memset(&var, 0, sizeof(struct Kokkos_Tools_VariableValue));
            // id
            std::getline(results, line);
            size_t id = atol(line.substr(line.find(delimiter)+2).c_str());
            var.type_id = id;
            // value
            std::getline(results, line);
            std::string value = line.substr(line.find(delimiter)+2);
            // get the variable name
            std::string varName = cachedVariableNames.find(id)->second;
            // look it up in the cached variables
            auto info = cachedVariables.find(varName);
            if (info->second.type == kokkos_value_double) {
                var.value.double_value = atof(value.c_str());
            } else if (info->second.type == kokkos_value_int64) {
                var.value.int_value = atol(value.c_str());
            } else {
                strcpy(var.value.string_value, value.c_str());
            }
            var.metadata = &(info->second);
            vars.insert(std::make_pair(id, std::move(var)));
        }
        cachedTunings.insert(std::make_pair(name, std::move(vars)));
    }
}

void KokkosSession::readCache(void) {
    std::ifstream results(cacheFilename);
    std::cout << "Reading cache of Kokkos tuning results from: '" << cacheFilename << "'" << std::endl;
    std::string line;
    while (std::getline(results, line))
    {
        std::istringstream iss(line);
        if (line.find("Input_", 0) != std::string::npos) {
            //std::cout << line << std::endl;
            parseVariableCache(results);
            continue;
        }
        if (line.find("Output_", 0) != std::string::npos) {
            //std::cout << line << std::endl;
            parseVariableCache(results);
            continue;
        }
        if (line.find("Context_", 0) != std::string::npos) {
            //std::cout << line << std::endl;
            parseContextCache(results);
            continue;
        }
    }
}

KokkosSession& KokkosSession::getSession() {
    static KokkosSession session;
    return session;
}

/* Have to make a deep copy of the variable to use it at exit */
void Variable::deepCopy(Kokkos_Tools_VariableInfo& _info) {
    info.type = _info.type;
    info.category = _info.category;
    info.valueQuantity = _info.valueQuantity;
    switch(info.category) {
        case kokkos_value_categorical:
        case kokkos_value_ordinal:
        {
            if (info.valueQuantity == kokkos_value_set) {
                size_t size = _info.candidates.set.size;
                info.candidates.set.size = size;
                if (info.type == kokkos_value_double) {
                    info.candidates.set.values.double_value =
                        (double*)(malloc(sizeof(double) * size));
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.set.values.int_value =
                        (int64_t*)(malloc(sizeof(int64_t) * size));
                } else if (info.type == kokkos_value_string) {
                    info.candidates.set.values.string_value =
                        (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String) * size));
                }
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        info.candidates.set.values.double_value[index] =
                            _info.candidates.set.values.double_value[index];
                    } else if (info.type == kokkos_value_int64) {
                        info.candidates.set.values.int_value[index] =
                            _info.candidates.set.values.int_value[index];
                    } else if (info.type == kokkos_value_string) {
                        //info.candidates.set.values.string_value[index] =
                        //    (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String)));
                        memcpy(&(info.candidates.set.values.string_value[index]),
                               &(_info.candidates.set.values.string_value[index]),
                               sizeof(Kokkos_Tools_Tuning_String));
                    }
                }
            }
            if (info.valueQuantity == kokkos_value_unbounded) {
            }
            break;
        }
        case kokkos_value_interval:
        case kokkos_value_ratio:
        {
            if (info.valueQuantity == kokkos_value_range) {
                if (info.type == kokkos_value_double) {
                    info.candidates.range.step.double_value =
                        _info.candidates.range.step.double_value;
                    info.candidates.range.lower.double_value =
                        _info.candidates.range.lower.double_value;
                    info.candidates.range.upper.double_value =
                        _info.candidates.range.upper.double_value;
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.range.step.int_value =
                        _info.candidates.range.step.int_value;
                    info.candidates.range.lower.int_value =
                        _info.candidates.range.lower.int_value;
                    info.candidates.range.upper.int_value =
                        _info.candidates.range.upper.int_value;
                }
                info.candidates.range.openLower = _info.candidates.range.openLower;
                info.candidates.range.openUpper = _info.candidates.range.openUpper;
            }
            if (info.valueQuantity == kokkos_value_set) {
                size_t size = _info.candidates.set.size;
                info.candidates.set.size = size;
                if (info.type == kokkos_value_double) {
                    info.candidates.set.values.double_value =
                        (double*)(malloc(sizeof(double) * size));
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.set.values.int_value =
                        (int64_t*)(malloc(sizeof(int64_t) * size));
                } else if (info.type == kokkos_value_string) {
                    info.candidates.set.values.string_value =
                        (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String) * size));
                }
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        info.candidates.set.values.double_value[index] =
                            _info.candidates.set.values.double_value[index];
                    } else if (info.type == kokkos_value_int64) {
                        info.candidates.set.values.int_value[index] =
                            _info.candidates.set.values.int_value[index];
                    } else if (info.type == kokkos_value_string) {
                        //info.candidates.set.values.string_value[index] =
                        //    (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String)));
                        memcpy(&(info.candidates.set.values.string_value[index]),
                               &(_info.candidates.set.values.string_value[index]),
                               sizeof(Kokkos_Tools_Tuning_String));
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

Variable::Variable(size_t _id, std::string _name,
    Kokkos_Tools_VariableInfo& _info) : id(_id), name(_name) {
        deepCopy(_info);
        // Create a hash object for strings
        std::hash<std::string> hasher;
        // Calculate the hash value of the string
        hashValue = std::to_string(hasher(name));
    /*
    if (KokkosSession::getSession().verbose) {
        std::cout << toString();
    }
    */
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
                    /*
                     * [] and () denote whether the range is inclusive/exclusive of the endpoint:
                     * [ includes the endpoint
                     * ( excludes the endpoint
                     * [] = 'Closed', includes both endpoints
                     * () = 'Open', excludes both endpoints
                     * [) and (] are both 'half-open', and include only one endpoint
                     */
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
    std::map<size_t, Variable*>& varmap, std::string tree_node) {
    std::stringstream ss;
    std::string d{"["};
    /* This is REALLY annoying. sometimes the order of the variables
     * can change, not sure how Kokkos is doing that but it may be a
     * side effect of using an unordered map. regardless, we have to
     * sort the variables by ID to make sure we generate the hash
     * consistently. */
    std::vector<std::pair<size_t,size_t>> reindex;
    for (size_t i = 0 ; i < numVars ; i++) {
        reindex.push_back(std::pair<size_t,size_t>(i,values[i].type_id));
    }
    sort(reindex.begin(), reindex.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.second < rhs.second;
            });
    for (size_t i = 0 ; i < numVars ; i++) {
        size_t ri = reindex[i].first;
        auto id = values[ri].type_id;
        Variable* var{varmap[id]};
        ss << d << var->name << ":";
        switch (var->info.type) {
            case kokkos_value_double:
                if (var->info.valueQuantity == kokkos_value_unbounded) {
                    ss << var->getBin(values[ri].value.double_value);
                } else {
                    ss << values[ri].value.double_value;
                }
                break;
            case kokkos_value_int64:
                if (var->info.valueQuantity == kokkos_value_unbounded) {
                    ss << var->getBin(values[ri].value.int_value);
                } else {
                    ss << values[ri].value.int_value;
                }
                break;
            case kokkos_value_string:
                ss << values[ri].value.string_value;
                break;
            default:
                break;
        }
        d = ",";
    }
    if(tree_node.size() > 0) {
        ss << ",tree_node:" << tree_node;
    }
    ss << "]";
    std::string tmp{ss.str()};
    return tmp;
}

void printContext(size_t numVars, std::string name) {
    std::cout << "-cv: " << numVars << name;
    std::cout << std::endl;
}

void printTuning(const size_t numVars, Kokkos_Tools_VariableValue* values,
    KokkosSession& session) {
    std::cout << "-tv: " << numVars;
    std::cout << hashContext(numVars, values, session.outputs, "");
    std::cout << std::endl;
}

bool getCachedTunings(std::string name,
    const size_t vars,
    Kokkos_Tools_VariableValue* values) {
    KokkosSession& session = KokkosSession::getSession();
    auto result = session.cachedTunings.find(name);
    // don't have a tuning for this context?
    if (result == session.cachedTunings.end()) { return false; }
    for (size_t i = 0 ; i < vars ; i++) {
        auto id = values[i].type_id;
        // look up the variable in the outputs, by id - may not match the cached ID
        auto outputVar = session.outputs.find(id);
        // look up the variable in the cache, by name - the name will always match
        std::string varname{outputVar->second->name};
        // look up the cached ID (it might not match the current variable id from Kokkos)
        size_t varID = session.cachedVariableIDs.find(varname)->second;
        auto variter = result->second.find(varID);
        //auto nameiter = session.cachedVariableNames.find(id);
        if (variter == result->second.end()) { return false; }
        //if (nameiter == session.cachedVariableNames.end()) { return false; }
        auto var = variter->second;
        //auto varname = nameiter->second;
        if (var.metadata->type == kokkos_value_double) {
            values[i].value.double_value = var.value.double_value;
            std::string tmp(name+":"+varname);
            apex::sample_value(tmp, var.value.double_value);
        } else if (var.metadata->type == kokkos_value_int64) {
            values[i].value.int_value = var.value.int_value;
            std::string tmp(name+":"+varname);
            apex::sample_value(tmp, var.value.int_value);
        } else if (var.metadata->type == kokkos_value_string) {
            strncpy(values[i].value.string_value, var.value.string_value, KOKKOS_TOOLS_TUNING_STRING_LENGTH);
        }
    }
    return true;
}

void set_params(std::shared_ptr<apex_tuning_request> request,
    const size_t vars,
    Kokkos_Tools_VariableValue* values) {
    for (size_t i = 0 ; i < vars ; i++) {
        auto id = values[i].type_id;
        Variable* var{KokkosSession::getSession().outputs[id]};
        if (var->info.valueQuantity == kokkos_value_set) {
            auto param = std::static_pointer_cast<apex_param_enum>(
                request->get_param(var->name));
            if (var->info.type == kokkos_value_double) {
                values[i].value.double_value = std::stod(param->get_value());
                std::string tmp(request->get_name()+":"+var->name);
                //if (!request->has_converged())
                    apex::sample_value(tmp, values[i].value.double_value);
            } else if (var->info.type == kokkos_value_int64) {
                values[i].value.int_value = std::stol(param->get_value());
                std::string tmp(request->get_name()+":"+var->name);
                //if (!request->has_converged())
                    apex::sample_value(tmp, values[i].value.int_value);
            } else if (var->info.type == kokkos_value_string) {
                strncpy(values[i].value.string_value, param->get_value().c_str(), KOKKOS_TOOLS_TUNING_STRING_LENGTH);
            }
        } else { // range
            if (var->info.type == kokkos_value_double) {
                auto param = std::static_pointer_cast<apex_param_double>(
                    request->get_param(var->name));
                values[i].value.double_value = param->get_value();
                std::string tmp(request->get_name()+":"+var->name);
                //if (!request->has_converged())
                    apex::sample_value(tmp, values[i].value.double_value);
            } else if (var->info.type == kokkos_value_int64) {
                auto param = std::static_pointer_cast<apex_param_long>(
                    request->get_param(var->name));
                values[i].value.int_value = param->get_value();
                std::string tmp(request->get_name()+":"+var->name);
                //if (!request->has_converged())
                    apex::sample_value(tmp, values[i].value.int_value);
            }
        }
    }
}

bool handle_start(const std::string & name, const size_t vars,
    Kokkos_Tools_VariableValue* values, uint64_t& delta, bool& converged) {
    KokkosSession& session = KokkosSession::getSession();
    auto search = session.requests.find(name);
    bool newSearch = false;
    if(search == session.requests.end()) {
        delta = apex::profiler::now_ns();
        // Start a new tuning session.
        if(session.verbose) {
            std::cout << std::string(getDepth(), ' ');
            std::cout << "Starting tuning session for " << name << std::endl;
        }
        std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
        session.requests.insert(std::make_pair(name, request));
        // save the variable ids associated with this session
        std::vector<int> var_ids;
        for (size_t i = 0 ; i < vars ; i++) {
            var_ids.push_back(values[i].type_id);
        }
        session.var_ids.insert(std::make_pair(name, var_ids));

        // Create an event to trigger this tuning session.
        apex_event_type trigger = apex::register_custom_event(name);
        request->set_trigger(trigger);

        // need this in the lambda
        bool verbose = session.verbose;
        // Create a metric
        std::function<double(void)> metric = [=]()->double{
            apex_profile * profile = apex::get_profile(name);
            if(profile == nullptr) {
                std::cerr << "ERROR: no profile for " << name << std::endl;
                //abort();
                return 0.0;
            }
            if(profile->calls == 0.0) {
                std::cerr << "ERROR: calls = 0 for " << name << std::endl;
                //abort();
                return 0.0;
            }
            double result = profile->minimum;
            if (result == 0.0) result = profile->accumulated/profile->calls;
            result = result * 1.0e-9; // convert to seconds to help search math
            if(verbose) {
                std::cout << std::string(getDepth(), ' ');
                std::cout << "querying time per call: " << result << "s" << std::endl;
            }
            return result;
        };
        request->set_metric(metric);

        // Set apex tuning strategy
        if (session.strategy == apex_ah_tuning_strategy::AUTOMATIC) {
            // just one variable?
            if (vars == 1) {
                auto id = values[0].type_id;
                Variable* var{session.outputs[id]};
                // and it's a small set of candidate values?
                if (var->info.valueQuantity == kokkos_value_set &&
                    var->info.candidates.set.size < 4) {
                    request->set_strategy(apex_ah_tuning_strategy::APEX_EXHAUSTIVE);
                // if integer, use simulated annealing
                } else if (var->info.type == kokkos_value_int64) {
                    request->set_strategy(apex_ah_tuning_strategy::SIMULATED_ANNEALING);
                // if double, use nelder mead
                } else if (var->info.type == kokkos_value_double) {
                    request->set_strategy(apex_ah_tuning_strategy::NELDER_MEAD_INTERNAL);
                }
            // more than one variable...
            } else {
                // are any of them categorical?
                bool haveSet = false;
                bool allDouble = true;
                for (size_t i = 0 ; i < vars ; i++) {
                    auto id = values[i].type_id;
                    Variable* var{session.outputs[id]};
                    if (var->info.valueQuantity == kokkos_value_set) {
                        haveSet = true;
                        allDouble = false;
                    } else if (var->info.type == kokkos_value_int64) {
                        allDouble = false;
                    }
                }
                // if have a categorical set, use genetic search
                if (haveSet) {
                    request->set_strategy(apex_ah_tuning_strategy::GENETIC_SEARCH);
                // if all double values, use nelder mead
                } else if (allDouble) {
                    request->set_strategy(apex_ah_tuning_strategy::NELDER_MEAD_INTERNAL);
                // as default, use simulated annealing
                } else {
                    request->set_strategy(apex_ah_tuning_strategy::SIMULATED_ANNEALING);
                }
            }
        } else {
            request->set_strategy(session.strategy);
        }
        request->set_radius(0.5);
        request->set_aggregation_times(3);
        // min, max, mean
        request->set_aggregation_function("min");

        for (size_t i = 0 ; i < vars ; i++) {
            auto id = values[i].type_id;
            Variable* var{session.outputs[id]};
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
                //printf("Initial string value: %s\n", front.c_str()); fflush(stdout);
                auto tmp = request->add_param_enum(
                    session.outputs[id]->name, front, space);
            } else {
                if (var->info.type == kokkos_value_double) {
                    double tval = values[i].value.double_value;
                    if (tval < session.outputs[id]->dmin ||
                        tval > session.outputs[id]->dmax) {
                        tval = session.outputs[id]->dmin;
                    }
                    auto tmp = request->add_param_double(
                        session.outputs[id]->name,
                        values[i].value.double_value,
                        session.outputs[id]->dmin,
                        session.outputs[id]->dmax,
                        session.outputs[id]->dstep);
                    //printf("Initial double value: %f\n", tval); fflush(stdout);
                } else if (var->info.type == kokkos_value_int64) {
                    int64_t tval = values[i].value.int_value;
                    if (tval < session.outputs[id]->lmin ||
                        tval > session.outputs[id]->lmax) {
                        tval = session.outputs[id]->lmin;
                    }
                    auto tmp = request->add_param_long(
                        session.outputs[id]->name,
                        tval,
                        session.outputs[id]->lmin,
                        session.outputs[id]->lmax,
                        session.outputs[id]->lstep);
                    //printf("Initial long value: %ld\n", tval); fflush(stdout);
                }
            }
        }

        // Start the tuning session.
        apex::setup_custom_tuning(*request);

        // Set OpenMP runtime parameters to initial values.
        set_params(request, vars, values);

        newSearch = true;
        // measure how long it took us to set this up
        delta = apex::profiler::now_ns() - delta;
    } else {
        // We've seen this region before.
        std::shared_ptr<apex_tuning_request> request = search->second;
        set_params(request, vars, values);
        converged = request->has_converged();
    }
    return newSearch;
}

void handle_stop(const std::string & name) {
    KokkosSession& session = KokkosSession::getSession();
    auto search = session.requests.find(name);
    /* We want to check if this search context has child contexts, and if so,
     * have they converged? If not converged, we want to pass in false. */
    bool childrenConverged = TreeNode::find(name, nullptr)->childrenConverged();
    if(search == session.requests.end()) {
        std::cerr << "ERROR: No data for " << name << std::endl;
    } else {
        apex_profile * profile = apex::get_profile(name);
        if(session.window == 1 ||
           (profile != nullptr &&
            profile->calls >= session.window)) {
            //std::cout << "Num calls: " << profile->calls << std::endl;
            std::shared_ptr<apex_tuning_request> request = search->second;
            /* If we are in a nested context, and this is the outermost
             * context, we want to not allow it to converge until all of
             * the inner contexts have also converged! */
            // Evaluate the results
            apex::custom_event(request->get_trigger(), &childrenConverged);
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
    if (!apex::apex_options::use_kokkos_tuning()) { return; }
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    KokkosSession& session = KokkosSession::getSession();
    session.checkForCache();
    if(session.verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __APEX_FUNCTION__ << " " << id << " " << info.type << "," << info.category << "," << info.valueQuantity << std::endl;
    }
    Variable * output = new Variable(id, name, info);
    output->makeSpace();
    session.saveOutputVar(id, output);
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
    if (!apex::apex_options::use_kokkos_tuning()) { return; }
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    KokkosSession& session = KokkosSession::getSession();
    session.checkForCache();
    if(session.verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __APEX_FUNCTION__ << " " << id << " " << info.type << "," << info.category << "," << info.valueQuantity << std::endl;
    }
    Variable * input = new Variable(id, name, info);
    session.saveInputVar(id, input);
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
    if (!apex::apex_options::use_kokkos_tuning()) { return; }
    // first, get the current timer node in the task tree
    //auto tlt = apex::thread_instance::get_top_level_timer();
    auto tlt = apex::thread_instance::instance().get_current_profiler();
    std::string tree_node{"default"};
    if (tlt != nullptr) {
        //tree_node = tlt->tt_ptr->tree_node->getName();
        tree_node = tlt->tt_ptr->task_id->get_name();
    }
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    KokkosSession& session = KokkosSession::getSession();
    // create a unique name for this combination of input vars
    std::string name{hashContext(numContextVariables, contextVariableValues,
        session.all_vars, tree_node)};
    if (session.verbose) {
        std::cout << std::string(getDepth(), ' ');
        std::cout << __APEX_FUNCTION__ << " ctx: " << contextId << std::endl;
        std::cout << std::string(getDepth(), ' ');
        printContext(numContextVariables, name);
    }
    // push our context on the stack
    if(session.contextStack.size() > 0) {
        session.contextStack.push(TreeNode::find(name, session.contextStack.top()));
    } else {
        session.contextStack.push(TreeNode::find(name, nullptr));
    }
    // check if we have a cached result
    bool success{false};
    if (session.use_history) {
        success = getCachedTunings(name, numTuningVariables, tuningVariableValues);
    }
    if (success) {
        session.used_history.insert(contextId);
    } else {
        uint64_t delta = 0;
        bool converged = false;
        if (handle_start(name, numTuningVariables, tuningVariableValues, delta, converged)) {
            // throw away the time spent setting up tuning
            //session.context_starts[contextId] = session.context_starts[contextId] + delta;
        }
        if(converged) {
            TreeNode::find(name, nullptr)->_hasConverged = true;
        }
        //if (!converged) {
            // add this name to our map of active contexts
            session.active_requests.insert(
                std::pair<uint32_t, std::string>(contextId, name));
        //}
    }
    if (session.verbose) {
        std::cout << std::string(getDepth(), ' ');
        printTuning(numTuningVariables, tuningVariableValues, session);
    }
}

/* This starts the context pointed at by contextId. If tools use
 * measurements to drive tuning, this is where they'll do their
 * starting measurement.
 */
void kokkosp_begin_context(size_t contextId) {
    if (!apex::apex_options::use_kokkos_tuning()) { return; }
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    KokkosSession& session = KokkosSession::getSession();
    if (session.verbose) {
        std::cout << std::string(getDepth()++, ' ');
        std::cout << __APEX_FUNCTION__ << "\t" << contextId << std::endl;
    }
    session.context_starts.insert(
        std::pair<uint32_t, uint64_t>(contextId, apex::profiler::now_ns()));
}

/* This simply says that the contextId in the argument is now over.
 * If you provided tuning values associated with that context, those
 * values can now be associated with a result.
 */
void kokkosp_end_context(const size_t contextId) {
    if (!apex::apex_options::use_kokkos_tuning()) { return; }
    // don't track memory in this function.
    apex::in_apex prevent_memory_tracking;
    KokkosSession& session = KokkosSession::getSession();
    uint64_t end = apex::profiler::now_ns();
    auto start = session.context_starts.find(contextId);
    auto name = session.active_requests.find(contextId);
    if (session.verbose) {
        std::cout << std::string(--getDepth(), ' ');
        std::cout << __APEX_FUNCTION__ << "\t" << contextId << std::endl;
    }
    if (name != session.active_requests.end() &&
        start != session.context_starts.end()) {
        if (session.verbose) {
            std::cout << std::string(getDepth(), ' ');
            std::cout << name->second << "\t" << (end-(start->second)) << " sec." << std::endl;
        }
        if (session.used_history.count(contextId) == 0) {
            apex::sample_value(name->second, (double)(end-(start->second)));
            handle_stop(name->second);
        } else {
            session.used_history.erase(contextId);
        }
        session.active_requests.erase(contextId);
    }
    session.context_starts.erase(contextId);
    if (session.contextStack.size() > 0) {
        session.contextStack.pop();
    }
}

} // extern "C"

