#include "exhaustive.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "apex_options.hpp"

namespace apex {

namespace exhaustive {

double inline myrand() {
    return ((double) rand() / (RAND_MAX));
}

size_t Exhaustive::get_max_iterations() {
    size_t max_iter{1};
    for (auto& v : vars) {
        switch (v.second.vtype) {
            case VariableType::doubletype: {
                max_iter = max_iter * v.second.dvalues.size();
                break;
            }
            case VariableType::longtype: {
                max_iter = max_iter * v.second.lvalues.size();
                break;
            }
            case VariableType::stringtype: {
                max_iter = max_iter * v.second.svalues.size();
                break;
            }
            default: {
                break;
            }
        }
    }
    // if we are only chosing between a couple of items, then
    // increase our number of iterations so that we ensure good
    // coverage.
    if (vars.size() == 1 || max_iter < 10) {
        max_iter = max_iter * std::max(10, apex_options::kokkos_tuning_window()+1);
    }
    // want to see multiple values of each one
    //std::cout << "Max iterations: " << max_iter << std::endl;
    return max_iter;
}

class log_wrapper {
    private:
        std::ofstream myfile;
    public:
        log_wrapper(const std::map<std::string, Variable>& vars) {
            myfile.open("tuning.csv");
            myfile << "iter,";
            for (auto& v : vars) { myfile << v.first << ","; }
            myfile << "time" << std::endl;
        }
        ~log_wrapper() {
            myfile.close();
        }
        std::ofstream& getstream() {
            return myfile;
        }
};

void Exhaustive::evaluate(double new_cost) {
    /*
    static log_wrapper log(vars);
    static size_t count{0};
    if (++count % 10000 == 0) { std::cout << count << std::endl; }
    log.getstream() << count << ",";
    for (auto& v : vars) { log.getstream() << v.second.toString() << ","; }
    log.getstream() << new_cost << std::endl;
    */
    if (new_cost < cost) {
        if (new_cost < best_cost) {
            best_cost = new_cost;
            if (apex_options::use_verbose()) {
                std::cout << "Exhaustive search: New best! " << new_cost << " k: " << k
                          << " kmax: " << kmax;
                for (auto& v : vars) {
                    std::cout  << ", " << v.first << ": " << v.second.toString();
                    v.second.save_best();
                }
            std::cout << std::endl;
            }
        }
        cost = new_cost;
    }
    //for (auto& v : vars) { v.second.choose_neighbor(); }
    k++;
    return;
}

} // exhaustive

} // apex


