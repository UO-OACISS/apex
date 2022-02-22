#include "exhaustive.hpp"
#include <algorithm>

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
    // want to see multiple values of each one
    return max_iter;
}

void Exhaustive::evaluate(double new_cost) {
    if (new_cost < cost) {
        if (new_cost < best_cost) {
            best_cost = new_cost;
            std::cout << "New best! " << new_cost << " k: " << k;
            for (auto& v : vars) { v.second.save_best(); }
            for (auto& v : vars) { std::cout  << ", index: " << v.second.current_index; }
            std::cout << std::endl;
        }
        cost = new_cost;
    }
    //for (auto& v : vars) { v.second.choose_neighbor(); }
    k++;
    return;
}

} // exhaustive

} // apex


