#pragma once

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace apex {
namespace internal {
namespace nelder_mead {

template <typename T> class Vec {
  public:
    Vec() {}
    Vec(unsigned int n) : n(n) {
        val.resize(n, 0);
    }
    Vec(std::initializer_list<T> c) {
        n = c.size();
        val.resize(n);
        std::copy(c.begin().c.end(), val.begin());
    }
    Vec(const Vec &lhs) {
        val = lhs.val;
        n   = lhs.n;
    }
    Vec(const std::vector<T> &lhs) {
        val = lhs;
        n   = lhs.size();
    }
    T operator()(unsigned int idx) const {
        if (idx >= n) {
            throw std::range_error("Element access out of range");
        }
        return val[idx];
    }
    T &operator()(unsigned int idx) {
        if (idx >= n) {
            throw std::range_error("Element access out of range");
        }
        return val[idx];
    }

    Vec operator=(const Vec &rhs) {
        val = rhs.val;
        n   = rhs.n;
        return *this;
    }

    Vec operator=(const std::vector<T> &rhs) {
        val = rhs;
        n   = rhs.size();
        return *this;
    }

    Vec operator+(const Vec &rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] + rhs.val[i];
        }
        return lhs;
    }
    Vec operator-(const Vec &rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] - rhs.val[i];
        }
        return lhs;
    }

    Vec operator/(T rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] / rhs;
        }
        return lhs;
    }
    Vec &operator+=(const Vec &rhs) {
        if (n != rhs.n)
            throw std::invalid_argument(
                "The two vectors must have the same length");
        for (unsigned int i = 0; i < n; i++) {
            val[i] += rhs.val[i];
        }
        return *this;
    }

    Vec &operator-=(const Vec &rhs) {
        if (n != rhs.n)
            throw std::invalid_argument(
                "The two vectors must have the same length");
        for (unsigned int i = 0; i < n; i++) {
            val[i] -= rhs.val[i];
        }
        return *this;
    }

    Vec &operator*=(T rhs) {
        for (unsigned int i = 0; i < n; i++) {
            val[i] *= rhs;
        }
        return *this;
    }

    Vec &operator/=(T rhs) {
        for (unsigned int i = 0; i < n; i++) {
            val[i] /= rhs;
        }
        return *this;
    }
    unsigned int size() const {
        return n;
    }
    unsigned int resize(unsigned int _n) {
        val.resize(_n);
        n = _n;
        return n;
    }
    T length() const {
        T ans = 0;
        for (unsigned int i = 0; i < n; i++) {
            ans += (val[i] * val[i]);
        }
        return std::sqrt(ans);
    }
    std::vector<T> &vec() {
        return val;
    }
    void enforce_min(std::vector<T> limit) {
        // don't exceed the limit!
        for (unsigned int i = 0; i < n; i++) {
            val[i] = std::max(val[i], limit[i]);
        }
    }
    void enforce_max(std::vector<T> limit) {
        // don't exceed the limit!
        for (unsigned int i = 0; i < n; i++) {
            val[i] = std::min(val[i], limit[i]);
        }
    }
    std::string to_string(void) {
        std::stringstream ss;
        ss << "[";
        for (unsigned int i = 0; i < n; i++) {
            ss << val[i];
            ss << (i == n - 1 ? "]" : ",");
        }
        return ss.str();
    }
    friend Vec operator*(T a, const Vec &b) {
        Vec c(b.size());
        for (unsigned int i = 0; i < b.size(); i++) {
            c.val[i] = a * b.val[i];
        }
        return c;
    }

  private:
    std::vector<T> val;
    unsigned int   n;
};

template <typename T = double> class Searcher {
    unsigned int                    dimension;
    bool                            adaptive;
    T                               tol_fun;
    T                               tol_x;
    unsigned int                    max_iter;
    unsigned int                    max_fun_evals;
    T                               alpha;
    T                               beta;
    T                               gamma;
    T                               delta;
    std::vector<Vec<T>>             simplex;
    unsigned int                    current_simplex_index;
    std::vector<std::pair<bool, T>> value_cache;
    unsigned int                    biggest_idx;
    unsigned int                    smallest_idx;
    T                               biggest_val;
    T                               second_biggest_val;
    T                               smallest_val;
    unsigned int                    niter;
    std::vector<T>                  res;
    unsigned int                    func_evals_count;
    std::vector<T>                  minimum_limits;
    std::vector<T>                  maximum_limits;
    Vec<T>                          x_bar; // centroid point
    Vec<T>                          x_r;   // reflection point
    Vec<T>                          x_e;   // expansion point
    Vec<T>                          x_c;   // contraction point
    T                               reflection_val;
    T                               expansion_val;
    T                               contraction_val;
    typedef enum {
        SIMPLEX,    // evaluating the simplex
        REFLECTION, // evaluate the reflection
        EXPANSION,  // evaluate the expansion
        CONTRACTION // evaluate the contraction
    } Stage_t;
    Stage_t stage;
    bool    _converged;
    bool    outside;

  public:
    bool verbose;
    Searcher(const std::vector<T> &initial_point,
             const std::vector<std::vector<T>> &initial_simplex = {},
             const std::vector<T> &_minimum_limits = {},
             const std::vector<T> &_maximum_limits = {}, bool _adaptive = true)
        : adaptive(_adaptive), tol_fun(1e-8), tol_x(1e-8), max_iter(1000),
          max_fun_evals(2000), current_simplex_index(0),
          minimum_limits(_minimum_limits), maximum_limits(_maximum_limits),
          _converged(false), verbose(false) {
        initialize(initial_point, initial_simplex);
    }
    void function_tolerance(T tol) {
        tol_fun = tol;
    }
    void point_tolerance(T tol) {
        tol_x = tol;
    }
    void initialize(const std::vector<T> &initial_point,
        const std::vector<std::vector<T>> &initial_simplex) {
        dimension = initial_point.size();
        assert(dimension > 0);
        // Setting parameters
        if (adaptive) {
            // Using the results of doi:10.1007/s10589-010-9329-3
            alpha = 1.0;
            beta  = 1.0 + (2.0 / dimension);
            gamma = 0.75 - (1.0 / (2.0 * dimension));
            delta = 1 - (1.0 / dimension);
        } else {
            alpha = 1.0;
            beta  = 2.0;
            gamma = 0.5;
            delta = 0.5;
        }
        //std::cout << alpha << " " << beta << " " << gamma << " " << delta
                  //<< std::endl;
        simplex.resize(dimension + 1);
        if (initial_simplex.empty()) {
            // Generate initial simplex
            simplex[0] = initial_point;
            for (unsigned int i = 1; i <= dimension; i++) {
                Vec<T> p(initial_point);
                T tau = (p(i - 1) < 1e-6 and p(i - 1) > -1e-6) ? 0.00025 : 0.05;
                p(i - 1) += tau;
                // enforce limits!
                if (minimum_limits.size() == dimension)
                    p.enforce_min(minimum_limits);
                if (maximum_limits.size() == dimension)
                    p.enforce_max(maximum_limits);
                simplex[i] = p;
            }
        } else {
            for (size_t i = 0 ; i < initial_simplex.size() ; i++) {
                simplex[i] = initial_simplex[i];
            }
        }
        value_cache.resize(dimension + 1);
        for (auto &v : value_cache) {
            v.first = false;
        }
        biggest_idx      = 0;
        smallest_idx     = 0;
        niter            = 0;
        func_evals_count = 0;
        stage            = SIMPLEX;
    }
    std::vector<T> &get_res(void) {
        return res;
    }
    const std::vector<T> &get_next_point(void) {
        if (converged())
            return res;
        switch (stage) {
        case SIMPLEX: {
            // return the next simplex point
            return simplex[current_simplex_index].vec();
        }
        case REFLECTION: {
            // Calculate the reflection point
            x_r = x_bar + alpha * (x_bar - simplex[biggest_idx]);
            // enforce limits!
            if (minimum_limits.size() == dimension)
                x_r.enforce_min(minimum_limits);
            if (maximum_limits.size() == dimension)
                x_r.enforce_max(maximum_limits);
            // return the reflection point
            return x_r.vec();
        }
        case EXPANSION: {
            // Calculate the Expansion point
            x_e = x_bar + beta * (x_r - x_bar);
            // enforce limits!
            if (minimum_limits.size() == dimension)
                x_e.enforce_min(minimum_limits);
            if (maximum_limits.size() == dimension)
                x_e.enforce_max(maximum_limits);
            // return the expansion point
            return x_e.vec();
        }
        case CONTRACTION: {
            // Compute the contraction point
            outside = false;
            // is the reflection better than the known worst?
            if (reflection_val < biggest_val) {
                // Outside contraction
                outside = true;
                x_c     = x_bar + gamma * (x_r - x_bar);
                // is the reflection worse than the known worst?
            } else if (reflection_val >= biggest_val) {
                // Inside contraction
                x_c = x_bar - gamma * (x_r - x_bar);
            }
            // return the contraction point
            return x_c.vec();
        }
        default: {
            assert(false);
        }
        }
        return res;
    }
    void report(T val) {
        if (converged())
            return;
        switch (stage) {
        case SIMPLEX:
            evaluate_simplex(val);
            break;
        case REFLECTION: // evaluate the reflection
            // assign the value to the current reflection point
            reflection_val = val;
            evaluate_reflection();
            break;
        case EXPANSION: // evaluate the expansion
            // assign the value to the current expansion point
            expansion_val = val;
            evaluate_expansion();
            break;
        case CONTRACTION: // evaluate the contraction
            // assign the value to the current contraction point
            contraction_val = val;
            evaluate_contraction();
            break;
        default:
            assert(false);
            break;
        }
        func_evals_count++;
        return;
    }
    bool converged(void) {
        return _converged;
    }
    void evaluate_simplex(T in_val) {
        // if we are still getting values for the simplex points, return
        if (current_simplex_index < value_cache.size()) {
            // assign the value to the current simplex point
            value_cache[current_simplex_index].first  = true;
            value_cache[current_simplex_index].second = in_val;
            if (verbose) {
                std::cout << "simplex " << current_simplex_index << " "
                          << simplex[current_simplex_index].to_string() << " = "
                          << in_val << std::endl;
            }
            current_simplex_index++;
            if (current_simplex_index < value_cache.size()) {
                return;
            }
        }
        // simplex populated, check for convergence and prep for reflection
        check_for_convergence();
    }

    void check_for_convergence(void) {
        // if we have values for all the simplex points, find the
        // best, worst, and second worst
        T val              = value_cache[0].second;
        biggest_val        = val;
        smallest_val       = val;
        second_biggest_val = val;
        biggest_idx        = 0;
        smallest_idx       = 0;
        // iterate over the other points in the simplex
        for (unsigned int i = 1; i < simplex.size(); i++) {
            // if we have a value, get it.
            val = value_cache[i].second;
            // is this value the biggest?
            if (val > biggest_val) {
                biggest_idx = i;
                biggest_val = val;
                // ...or is it the smallest?
            } else if (val < smallest_val) {
                smallest_idx = i;
                smallest_val = val;
            }
        }
        // at this point, we should know the biggest and smallest value
        // and the simplexes that generated them

        // Calculate the difference of function values and the distance between
        // points in the simplex, so that we can know when to stop the
        // optimization
        T max_val_diff   = 0;
        T max_point_diff = 0;
        for (unsigned int i = 0; i < simplex.size(); i++) {
            val = value_cache[i].second;
            // find the second biggest value, save it for later reflection
            if (i != biggest_idx and val > second_biggest_val) {
                second_biggest_val = val;
                // is this NOT the smallest value in the current set of simplex
                // points?
            } else if (i != smallest_idx) {
                // how far is this point from the current best, and is it the
                // furthest point from the best?
                if (std::abs(val - smallest_val) > max_val_diff) {
                    max_val_diff = std::abs(val - smallest_val);
                }
                // get the manhattan distance of the vector between this point
                // and the smallest one we have seen so far
                T diff = (simplex[i] - simplex[smallest_idx]).length();
                // how "far" is the point from the smallest point?
                if (diff > max_point_diff) {
                    max_point_diff = diff;
                }
            }
        }
        // have we converged? either by being within tolerance of the best -
        // worst or by being within the tolerance of a point distance?
        if ((max_val_diff <= tol_fun and max_point_diff <= tol_x) or
            (func_evals_count >= max_fun_evals) or (niter >= max_iter)) {
            res = simplex[smallest_idx].vec();
            std::cout << "Converged after " << niter << " iterations."
                      << std::endl;
            std::cout << "Total func evaluations: " << func_evals_count
                      << std::endl;
            _converged = true;
            return;
        } else if (verbose) {
            std::cout << "Not converged: " << max_val_diff << " value difference."
                      << std::endl;
            std::cout << "Not converged: " << max_point_diff << " point difference."
                      << std::endl;
        }

        // not converged?
        // Calculate the centroid of our current set
        x_bar.resize(dimension);
        for (unsigned int i = 0; i < dimension; i++)
            x_bar(i) = 0;
        for (unsigned int i = 0; i < simplex.size(); i++) {
            if (i != biggest_idx)
                x_bar += simplex[i];
        }
        x_bar /= dimension;

        // advance to REFLECTION
        stage = REFLECTION;
        return;
    }
    void evaluate_reflection(void) {
        niter++;
        if (verbose) {
            std::cout << "reflection = " << x_r.to_string() << " "
                      << reflection_val << std::endl;
        }
        // is it better than our current best?
        if (reflection_val < smallest_val) {
            stage = EXPANSION;
            return;
        } else if (reflection_val >= second_biggest_val) {
            stage = CONTRACTION;
            return;
        } else {
            // Reflection is good enough
            // replace the worst point with the reflection point.
            simplex[biggest_idx]            = x_r;
            value_cache[biggest_idx].second = reflection_val;
            // simplex populated, check for convergence and prep for reflection
            check_for_convergence();
        }
        // reflection fell through, so stay in reflection stage - but get a new
        // point
        stage = REFLECTION;
    }
    void evaluate_expansion(void) {
        // evaluate the expansion point
        if (verbose) {
            std::cout << "expansion = " << x_e.to_string() << " "
                      << expansion_val << std::endl;
        }
        // is the expansion point better than the reflection point?
        if (expansion_val < reflection_val) {
            // replace the worst simplex point with our new expansion point
            simplex[biggest_idx]            = x_e;
            value_cache[biggest_idx].second = expansion_val;
            // simplex populated, check for convergence and prep for reflection
            check_for_convergence();
        } else {
            // otherwise, replace our worst simplex with our reflection point
            simplex[biggest_idx]            = x_r;
            value_cache[biggest_idx].second = reflection_val;
            // simplex populated, check for convergence and prep for reflection
            check_for_convergence();
        }
        if (current_simplex_index > dimension) {
            stage = REFLECTION;
        } else {
            // stage = SIMPLEX;
            stage = REFLECTION;
        }
    }
    void evaluate_contraction(void) {
        // evaluate the contraction point
        if (verbose) {
            std::cout << "contraction = " << x_c.to_string() << " "
                      << contraction_val << std::endl;
        }
        // is the contraction better than the reflection or the known worst?
        if ((outside and contraction_val <= reflection_val) or
            (not outside and contraction_val <= biggest_val)) {
            // replace the known worst with the contraction
            simplex[biggest_idx]            = x_c;
            value_cache[biggest_idx].second = contraction_val;
            // simplex populated, check for convergence and prep for reflection
            check_for_convergence();
            stage = REFLECTION;
        } else {
            // Shrinking
            if (verbose) {
                std::cout << "shrinking, smallest index = " << smallest_idx << std::endl;
            }
            // we take the whole simplex, and move every point towards the
            // current best candidate
            for (unsigned int i = 0; i < dimension; i++) {
                if (i != smallest_idx) {
                    simplex[i] = simplex[smallest_idx] +
                                 delta * (simplex[i] - simplex[smallest_idx]);
                    value_cache[i].first = false;
                }
            }
            // have to re-evaluate the new simplex, so reset.
            current_simplex_index = 0;
            stage                 = SIMPLEX;
        }
    }
};

}; // namespace nelder_mead
}; // namespace internal
}; // namespace apex
