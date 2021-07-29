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

namespace simulated_annealing {

enum class VariableType { doubletype, longtype, stringtype } ;

/* Original, wored at one point
int inline myrandn() {
    static std::default_random_engine generator;
    static std::normal_distribution<double> distribution(0.0,2.0);
    return (int)(std::round(distribution(generator)));
}
*/

/* not sure... pretty random!
*/
double inline myrandn() {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(-1.0,1.0);
    return distribution(generator);
}

/* just integers from -5, to 5
int inline myrandn() {
    static std::default_random_engine generator;
    static std::uniform_int_distribution<int> distribution(-10,10);
    return distribution(generator);
}
*/

class Variable {
public:
    std::vector<double> dvalues;
    std::vector<long> lvalues;
    std::vector<std::string> svalues;
    VariableType vtype;
    size_t current_index;
    size_t neighbor_index;
    size_t best_index;
    void * value; // for the client to get the values
    size_t maxlen;
    size_t half;
    double quarter;
    Variable () = delete;
    Variable (VariableType vtype, void * ptr) : vtype(vtype), current_index(0),
        neighbor_index(0), best_index(0), value(ptr), maxlen(0) { }
    void get_random_neighbor(double scope) {
        APEX_UNUSED(scope);
        //int delta = myrandn(half*scope);
        //int delta = (int)(myrandn() * quarter * scope);
        int delta = (int)(myrandn() * quarter * scope);
        //int delta = myrandn();
        //printf("Trying %d...", delta);
        if (delta < 0 && (current_index < (size_t)(abs(delta)))) {
            // do nothing
            //neighbor_index = 0;
        } else if (delta > 0 && ((current_index + delta) > maxlen)) {
            // do nothing
            //neighbor_index = maxlen;
        } else {
            neighbor_index = current_index + delta;
        }
        if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[neighbor_index];
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[neighbor_index];
        }
        else {
            *((const char**)(value)) = svalues[neighbor_index].c_str();
        }
    }
    void choose_neighbor() { current_index = neighbor_index; }
    void save_best() { best_index = current_index; }
    void restore_best() { current_index = best_index; }
    /* For initializing in the center of the space */
    void set_init() {
        maxlen = (std::max(std::max(dvalues.size(),
            lvalues.size()), svalues.size())) - 1;
        half = maxlen/2;
        quarter = (double)half/2;
        current_index = neighbor_index = best_index = half;
        //std::cout << "Initialized to " << current_index << std::endl;
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
};

/* In the formulation of the method by Kirkpatrick et al., the acceptance
 * probability function P(e,e',T) was defined as 1 if e'<e, and
 * exp(-(e'-e)/T) otherwise.
 */
/*
 * Let s = s0
 * For k = 0 through kmax (exclusive):
 *   T <- temperature( (k+1)/kmax )
 *   Pick a random neighbour, snew <- neighbour(s)
 *   If P(E(s), E(snew), T) ≥ random(0, 1):
 *     s <- snew
 * Output: the final state s
 */

class SimulatedAnnealing {
private:
    double cost;
    double best_cost;
    size_t restart;
    size_t since_restart;
    double temp;
    size_t kmax;
    size_t k;
    std::map<std::string, Variable> vars;
public:
    void evaluate(double new_cost);
    SimulatedAnnealing() :
        restart(0), since_restart(0), temp(0), kmax(0), k(1) {
        cost = std::numeric_limits<double>::max();
        best_cost = cost;
    }
    double getEnergy() { return best_cost; }
    bool converged() {
        return (k >= kmax);
    }
    void getNewSettings() {
        /*   Pick a random neighbour, snew <- neighbour(s) */
        for (auto& v : vars) { v.second.get_random_neighbor(1-temp); }
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
    double acceptance_probability(double new_cost);
    size_t get_max_iterations();
    std::map<std::string, Variable>& get_vars() { return vars; }
    void add_var(std::string name, Variable var) {
        vars.insert(std::make_pair(name, var));
        kmax = get_max_iterations();
        /* get max iterations */
        //std::cout << "Max iterations : " << kmax << std::endl;
        restart = kmax / 10;
    }
};

} // simulated_annealing

} // apex
