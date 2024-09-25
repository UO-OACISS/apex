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

namespace apex {

namespace genetic {

enum class VariableType { doubletype, longtype, stringtype } ;

class Variable {
public:
    std::vector<double> dvalues;
    std::vector<long> lvalues;
    std::vector<std::string> svalues;
    VariableType vtype;
    size_t current_index;
    size_t best_index;
    void * value; // for the client to get the values
    size_t maxlen;
    Variable () = delete;
    Variable (VariableType vtype, void * ptr) : vtype(vtype), current_index(0),
        best_index(0), value(ptr), maxlen(0) { }
    void set_current_value() {
        if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[current_index];
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[current_index];
        }
        else {
            *((const char**)(value)) = svalues[current_index].c_str();
        }
    }
    size_t get_next_neighbor() {
        current_index = (rand() % maxlen);
        APEX_ASSERT(current_index < maxlen);
        set_current_value();
        return current_index;
    }
    void save_best() { best_index = current_index; }
    void set_init(double init_value) {
        auto it = std::find(dvalues.begin(), dvalues.end(), init_value);
        current_index = distance(dvalues.begin(), it);
        set_current_value();
    }
    void set_init(long init_value) {
        auto it = std::find(lvalues.begin(), lvalues.end(), init_value);
        current_index = distance(lvalues.begin(), it);
        set_current_value();
    }
    void set_init(std::string init_value) {
        auto it = std::find(svalues.begin(), svalues.end(), init_value);
        current_index = distance(svalues.begin(), it);
        set_current_value();
    }
    std::string getBest() {
        if (vtype == VariableType::doubletype) {
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
};

struct individual {
    std::vector<size_t> indexes;
    double cost;
};

class GeneticSearch {
private:
    double cost;
    double best_cost;
    size_t kmax;
    size_t k;
    std::map<std::string, Variable> vars;
    const size_t max_iterations{500};
    const size_t min_iterations{16};
    const size_t population_size{32};
    const size_t crossover{16}; // half population
    const size_t parent_ratio{50};
    const size_t mutate_probability{15};
    const size_t max_stable_generations{5};
    size_t num_stable_generations;
    std::vector<individual> population;
    bool bootstrapping;
    double best_generation_cost;
public:
    void evaluate(double new_cost);
    GeneticSearch() :
        kmax(0), k(1), num_stable_generations(0), bootstrapping(true),
        best_generation_cost(0.0) {
        cost = std::numeric_limits<double>::max();
        best_cost = cost;
        //std::cout << "New Session!" << std::endl;
        //srand (1);
        srand (time(NULL));
    }
    bool converged() {
        // if we haven't improved for X generations, quit
        if (num_stable_generations >= max_stable_generations) {
            return true;
        }
        // otherwise, just quit when we hit max iterations
        return (k > kmax);
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
        kmax = get_max_iterations();
        /* get max iterations */
        //std::cout << "Max iterations : " << kmax << std::endl;
    }
};

} // genetic

} // apex
