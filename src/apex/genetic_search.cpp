#include "apex.hpp"
#include "genetic_search.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

namespace apex {

namespace genetic {

double inline myrand() {
    return ((double) rand() / (RAND_MAX));
}

size_t GeneticSearch::get_max_iterations() {
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
    //return max_iter;
    return std::min(max_iterations, (std::max(min_iterations, max_iter)));
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

/* OK, here's where ALL the book keeping is going to happen.
   The basic approach is to do the following:
    1. Create N initial random generation combinations
    2. Evaluate all N individuals
    3. Sort individuals
    4. Drop bottom 1/2 of individuals
    5. Replace dropped individuals with "genes" crossed from each higher ranked
       individuals, probabilities proportional to their rankings. That is, the
       highest ranked individual is most likely to contribute their genes to
       the next generation of individuals.
    6. mutate some of the new individuals accordning to some probability
    7. Go back to 2, iterate until we don't get a new highest ranked
       individual for X generations.

    Constants needed:
        population size (N): probably some power of 2, dependent on the number of variables?
        parent ratio
        mutate probability
        transfer ratio
        crossover
*/

auto get_random_number(const std::size_t min, const std::size_t max) {
    const std::size_t values_count = max - min + 1;
    return rand() % values_count + min;
}

auto get_rank_order(const std::size_t min, const std::size_t max) {
    // get 2^max minus 1
    size_t powmax = pow(2,max);
    size_t rando = get_random_number(min,powmax);
    size_t index = 0;
    for (size_t i = pow(2,max-1) ; i > min ; i = i / 2) {
        if (rando > i) { break; }
        index++;
    }
    return index;
}

void GeneticSearch::getNewSettings() {
    if (bootstrapping) {
        if (population.size() >= population_size) {
            bootstrapping = false;
        } else {
            // we are still bootstrapping, so just get a random selection.
            for (auto& v : vars) { v.second.get_next_neighbor(); }
            best_generation_cost = best_cost;
            return;
        }
    }
    //std::cout << "Have population of " << population.size() << " to evaluate!" << std::endl;
    // time to cull the herd?
    if (population.size() >= population_size) {
        if (best_cost < best_generation_cost) {
            best_generation_cost = best_cost;
        } else {
            num_stable_generations++;
        }
        // we need to sort the population...
        sort(population.begin(), population.end(),
            [](const individual& lhs, const individual& rhs) {
                return lhs.cost < rhs.cost;
            });
        // ...then drop half of them - the "weakest" ones.
        population.erase(population.cbegin() + crossover, population.cend());
        //std::cout << "Now have population of " << population.size() << std::endl;
    }
    // We want to generate a new individual using two "high quality" parents.
    // choose parent A
    auto indexA = get_rank_order(0,crossover-1);
    individual& A = population[indexA];
    /*
    std::cout << "A: " << indexA;
    for (auto& i : A.indexes) { std::cout << "," << i; }
    std::cout << ", " << A.cost << std::endl;
    */
    // choose parent B
    auto indexB = get_rank_order(0,crossover-1);
    individual& B = population[indexB];
    /*
    std::cout << "B: " << indexB;
    for (auto& i : B.indexes) { std::cout << "," << i; }
    std::cout << ", " << B.cost << std::endl;
    */
    // blend their variables into a new individual and maybe mutate?
    size_t i = 0;
    for (auto& v : vars) {
        // if mutating, just get a random value.
        if (get_random_number(0,100) < mutate_probability) {
            v.second.get_next_neighbor();
        // otherwise, get a "gene" from a parent
        } else if (get_random_number(0,100) < parent_ratio) {
            v.second.current_index = A.indexes[i];
        } else {
            v.second.current_index = B.indexes[i];
        }
        i++;
    }
}

void GeneticSearch::evaluate(double new_cost) {
    /*
    static log_wrapper log(vars);
    log.getstream() << k << ",";
    for (auto& v : vars) { log.getstream() << v.second.toString() << ","; }
    log.getstream() << new_cost << std::endl;
    */
    if (new_cost < cost) {
        if (new_cost < best_cost) {
            best_cost = new_cost;
            if (apex_options::use_verbose()) {
                std::cout << "Genetic Search: New best! " << new_cost << " k: " << k
                          << " kmax: " << kmax;
                for (auto& v : vars) {
                    std::cout  << ", " << v.first << ": " << v.second.toString();
                    v.second.save_best();
                }
                std::cout << std::endl;
            }
        }
        cost = new_cost;
        /*
    } else {
        std::cout << "          " << new_cost << " k: " << k;
        for (auto& v : vars) { std::cout  << ", value: " << v.second.toString(); }
        std::cout << std::endl;
        */
    }
    /* save our individual in the population */
    individual i;
    i.cost = new_cost;
    for (auto& v : vars) { i.indexes.push_back(v.second.current_index); }
    population.push_back(i);
    k++;
    return;
}

} // genetic

} // apex


