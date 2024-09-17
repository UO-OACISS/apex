#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <map>
#include "apex_types.h"
#include "apex_assert.h"
#include "nelder_mead_internal.h"

namespace apex {

namespace nelder_mead {

enum class VariableType { doubletype, longtype, stringtype, continuous } ;

class Variable {
public:
    std::vector<double> dvalues;
    std::vector<long> lvalues;
    std::vector<std::string> svalues;
    std::array<double,2> limits;
    VariableType vtype;
    size_t current_index;
    size_t best_index;
    double current_value;
    double best_value;
    void * value; // for the client to get the values
    size_t maxlen;
    Variable () = delete;
    Variable (VariableType vtype, void * ptr) : vtype(vtype), current_index(0),
        best_index(0), current_value(0), best_value(0), value(ptr), maxlen(0) { }
    void set_current_value() {
        if (vtype == VariableType::continuous) {
            *((double*)(value)) = current_value;
        }
        else if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[current_index];
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[current_index];
        }
        else {
            *((const char**)(value)) = svalues[current_index].c_str();
        }
    }
    void save_best() {
        if (vtype == VariableType::continuous) {
            best_value = current_value;
        } else {
            best_index = current_index;
        }
    }
    void set_init() {
        maxlen = (std::max(std::max(dvalues.size(),
            lvalues.size()), svalues.size()));
        current_index = 0;
        set_current_value();
    }
    std::string getBest() {
        if (vtype == VariableType::continuous) {
            *((double*)(value)) = best_value;
            return std::to_string(best_value);
        }
        else if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[best_index];
            return std::to_string(dvalues[best_index]);
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[best_index];
            return std::to_string(lvalues[best_index]);
        }
        //else if (vtype == VariableType::stringtype) {
        *((const char**)(value)) = svalues[best_index].c_str();
        return svalues[best_index];
    }
    std::string toString() {
        if (vtype == VariableType::continuous) {
            return std::to_string(current_value);
        }
        if (vtype == VariableType::doubletype) {
            return std::to_string(dvalues[current_index]);
        }
        else if (vtype == VariableType::longtype) {
            return std::to_string(lvalues[current_index]);
        }
        //else if (vtype == VariableType::stringtype) {
        return svalues[current_index];
        //}
    }
    double get_init(void) {
        if (vtype == VariableType::continuous) {
            // choose some point in the middle between the limits
            return (dvalues[0] + dvalues[1]) / 2.0;
        }
        // otherwise, choose an index somewhere in the middle
        //return ((double)maxlen / 2.0);
        return 0.5;
    }
    const std::array<double,2>& get_limits(void) {
        // if our variable is continuous, we have been initialized with
        // two values, the min and the max
        if (vtype == VariableType::continuous) {
            limits[0] = dvalues[0];
            limits[1] = dvalues[1];
        // if our variable is discrete, we will use the range from 0 to 1,
        // and scale that value to the number of descrete values we have to get
        // an index.
        } else {
            limits[0] = 0.0;
            limits[1] = 1.0;
        }
        return limits;
    }
};

class NelderMead {
private:
    double cost;
    double best_cost;
    size_t kmax;
    size_t k;
    std::map<std::string, Variable> vars;
    const size_t max_iterations{500};
    const size_t min_iterations{16};
    internal::nelder_mead::Searcher<double>* searcher;
    bool hasDiscrete;
public:
    void evaluate(double new_cost);
    NelderMead() :
        kmax(0), k(1), searcher(nullptr), hasDiscrete(false) {
        cost = std::numeric_limits<double>::max();
        best_cost = cost;
    }
    ~NelderMead() {
        if (searcher != nullptr) {
            delete searcher;
        }
    }
    bool converged() {
        return searcher->converged();
    }
    void getNewSettings();
    void saveBestSettings() {
        for (auto& v : vars) { v.second.getBest(); }
    }
    void printBestSettings() {
        std::string d("[");
        for (auto v : vars) {
            std::cout << d << v.second.getBest();
            d = ",";
        }
        std::cout << "]" << std::endl;
    }
    size_t get_max_iterations();
    std::map<std::string, Variable>& get_vars() { return vars; }
    void add_var(std::string name, Variable var) {
        vars.insert(std::make_pair(name, var));
        if (var.vtype == VariableType::longtype ||
            var.vtype == VariableType::stringtype) {
            hasDiscrete = true;
        }
    }
    void start(void);
};

} // nelder_mead

} // apex
