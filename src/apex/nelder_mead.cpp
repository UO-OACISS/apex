#include "nelder_mead.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

namespace apex {

namespace nelder_mead {

void NelderMead::start(void) {
    // create a starting point
    std::vector<double> init_point;
    for (auto& v : vars) {
        init_point.push_back(v.second.get_init());
    }
    // create a lower limit
    std::vector<double> lower_limit;
    std::vector<double> upper_limit;
    for (auto& v : vars) {
        auto& limits = v.second.get_limits();
        lower_limit.push_back(limits[0]);
        upper_limit.push_back(limits[1]);
    }
    // create a starting simplex - random values in the space, nvars+1 of them
    std::vector<std::vector<double>> init_simplex;
    for (size_t i = 0 ; i < (vars.size() + 1) ; i++) {
        std::vector<double> tmp;
        for (auto& v : vars) {
            double r = ((double) std::rand() / (RAND_MAX));
            auto& limits = v.second.get_limits();
            double range = limits[1] - limits[0];
            double sample_in_range = range * r;
            tmp.push_back(limits[0] + sample_in_range);
        }
        //std::cout << "range: [" << lower_limit[i] << "," << upper_limit[i] << "] value: ["
            //<< tmp[0] << "," << tmp[1] << "]" << std::endl;
        init_simplex.push_back(tmp);
    }
    searcher = new apex::internal::nelder_mead::Searcher<double>(init_point, init_simplex, lower_limit, upper_limit, true);
    searcher->function_tolerance(10000);
    if (hasDiscrete) {
        searcher->point_tolerance(1.0);
    } else {
        searcher->point_tolerance(0.01);
    }
}

/*
static std::string vector_to_string(std::vector<double> val) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < val.size(); i++) {
        ss << val[i];
        ss << (i == val.size() - 1 ? "]" : ",");
    }
    return ss.str();
}
*/

void NelderMead::getNewSettings() {
    if (searcher == nullptr) start();
    // get the next point from the simplex search
    auto point = searcher->get_next_point();
    //std::cout << "Next point: " << vector_to_string(point) << std::endl;
    size_t i{0};
    for (auto& v : vars) {
        // if continuous, we just get the value
        if (v.second.vtype == VariableType::continuous) {
            v.second.current_value = point[i];
        } else {
            // otherwise, we scale the value from [0:maxlen] to our discrete index
            double tmp = point[i] * v.second.maxlen;
            v.second.current_index = std::min((size_t)(std::trunc(tmp)),(v.second.maxlen-1));
        }
        v.second.set_current_value();
        i++;
    }
}

void NelderMead::evaluate(double new_cost) {
    // report the result
    searcher->report(new_cost);
    if (new_cost < cost) {
        if (new_cost < best_cost) {
            best_cost = new_cost;
            std::cout << "Nelder Mead: New best! " << new_cost << " k: " << k;
            for (auto& v : vars) {
                std::cout  << ", " << v.first << ": " << v.second.toString();
                v.second.save_best();
            }
            std::cout << std::endl;
        }
        cost = new_cost;
        // if the function evaluation takes a long time (in nanoseconds, remember), increase our tolerance.
        auto tmp = std::max((new_cost / 50.0), 1000.0);
        //std::cout << "new function tolerance: " << tmp << std::endl;
        searcher->function_tolerance(tmp);
    }
    k++;
    return;
}

} // genetic

} // apex


