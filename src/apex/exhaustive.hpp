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

namespace apex {

namespace exhaustive {

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
    size_t max_index;
    Variable () = delete;
    Variable (VariableType vtype, void * ptr) : vtype(vtype), current_index(0),
        best_index(0), value(ptr), max_index(0) { }
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
        current_index++;
        if (current_index > max_index) {
            current_index = 0;
        }
        set_current_value();
        return current_index;
    }
    void save_best() { best_index = current_index; }
    void set_init() {
        max_index = (std::max(std::max(dvalues.size(),
            lvalues.size()), svalues.size())) - 1;
        current_index = max_index;
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

class Exhaustive {
private:
    double cost;
    double best_cost;
    size_t kmax;
    size_t k;
    std::map<std::string, Variable> vars;
    //const size_t max_iterations{1000};
    //const size_t min_iterations{100};
public:
    void evaluate(double new_cost);
    Exhaustive() :
        kmax(0), k(1) {
        cost = std::numeric_limits<double>::max();
        best_cost = cost;
        //std::cout << "New Session!" << std::endl;
    }
    double getEnergy() { return best_cost; }
    bool converged() { return (k >= kmax); }
    void getNewSettings() {
        /*   Increment neighbour */
        for (auto& v : vars) {
            size_t index = v.second.get_next_neighbor();
            if (index != 0) {
                break;
            }
        }
    }
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

} // exhaustive

} // apex
