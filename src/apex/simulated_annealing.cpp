#include "simulated_annealing.hpp"

namespace apex {

namespace simulated_annealing {

double inline myrand() {
    return ((double) rand() / (RAND_MAX));
}

size_t SimulatedAnnealing::get_max_iterations() {
    size_t max_iter{1};
    for (auto& v : vars) {
        switch (v.second.vtype) {
            case VariableType::doubletype: {
                max_iter = max_iter + v.second.dvalues.size();
                break;
            }
            case VariableType::longtype: {
                max_iter = max_iter + v.second.lvalues.size();
                break;
            }
            case VariableType::stringtype: {
                max_iter = max_iter + v.second.svalues.size();
                break;
            }
            default: {
                break;
            }
        }
    }
    //return max_iter / vars.size();
    //return max_iter * vars.size() *vars.size();
    return max_iter / 3; // the window
}

double SimulatedAnnealing::acceptance_probability(double new_cost) {
    if (new_cost < cost) {
        return 1.0;
    }
    return exp(-1.0 * ((new_cost-cost) / temp));
}

void SimulatedAnnealing::evaluate(double new_cost) {
    /*   T <- temperature( (k+1)/kmax ) */
    temp = (double)(k)/(double)(kmax);
    /*   If P(E(s), E(snew), T) ≥ random(0, 1): */
    /*     s <- snew */
    if (new_cost < cost) {
        for (auto& v : vars) { v.second.choose_neighbor(); }
        if (new_cost < best_cost) {
            best_cost = new_cost;
            std::cout << "New best! " << new_cost << " k: " << k << " temp: " << temp;
            for (auto& v : vars) { v.second.save_best(); }
            for (auto& v : vars) { std::cout  << ", index: " << v.second.current_index; }
            std::cout << std::endl;
            since_restart = 1;
        }
        cost = new_cost;
    } else {
        double pE{acceptance_probability(new_cost)};
        double pRand{myrand()};
        //std::cout << pE << " >= " <<pRand << std::endl;
        if (pE >= pRand) {
            for (auto& v : vars) { v.second.choose_neighbor(); }
            cost = new_cost;
        }
    }
    /* Check if we need to restart the search at last known best */
    if (since_restart % restart == 0) {
        //std::cout << "Restarting to last known good" << std::endl;
        cost = best_cost;
        for (auto& v : vars) { v.second.restore_best(); }
        since_restart = 1;
    }
    since_restart++;
    k++;
    return;
}

} // simulated_annealing

} // apex

#if 0
void make_init_vars(SimulatedAnnealing& request) {
    double dmin = 0.00;
    double dmax = 1.0;
    double dstep = 0.06;
    Variable var(VariableType::doubletype);
    double dvalue = dmin;
    do {
        var.dvalues.push_back(dvalue);
        dvalue = dvalue + dstep;
    } while (dvalue < dmax);
    var.set_init();
    request.add_var("double", std::move(var));

    size_t lmin = 1;
    size_t lmax = 128;
    Variable var2(VariableType::longtype);
    size_t lvalue = lmin;
    do {
        //std::cout << "Pushing " << lvalue << std::endl;
        var2.lvalues.push_back(lvalue);
        lvalue = lvalue * 2;
    } while (lvalue < lmax);
    var2.set_init();
    request.add_var("long", std::move(var2));

    size_t smin = 1;
    size_t smax = 32;
    Variable var3(VariableType::stringtype);
    size_t svalue = smin;
    do {
        //std::cout << "Pushing " << svalue << std::endl;
        std::string tmp(svalue, 'a');
        var3.svalues.push_back(tmp);
        svalue++;
    } while (svalue < smax);
    var3.set_init();
    request.add_var("string", std::move(var3));
}

double get_new_cost(std::map<std::string, Variable>& vars) {
    double cost{0.0};
    for (auto& v : vars) {
        if (v.second.vtype == VariableType::doubletype) {
            cost += v.second.dvalues[v.second.neighbor_index];
        }
        else if (v.second.vtype == VariableType::longtype) {
            cost += (double)(128 - v.second.lvalues[v.second.neighbor_index]);
        }
        else if (v.second.vtype == VariableType::stringtype) {
            cost += (double)(16 - v.second.svalues[v.second.neighbor_index].size());
        }
    }
    /*
    std::cout << "new cost: " << cost;
    for (Variable& v : vars) {
        std::cout << "," << v.neighbor_index;
    }
    std::cout << std::endl;
    */
    return cost;
}

int main (int argc, const char * argv[]) {
    SimulatedAnnealing request;
    // make variables
    make_init_vars(request);
    while (!request.converged()) {
        request.getNewSettings();
        double new_cost{get_new_cost(request.get_vars())};
        request.evaluate(new_cost);
    }
    double e = request.getEnergy();
    std::cout << "Final cost: " << e;
    std::string d{": "};
    for (auto v : request.get_vars()) {
        std::cout << d << v.second.getBest();
        d = ",";
    }
    std::cout << std::endl;
}

#endif

