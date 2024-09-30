#include <tuning_playground.hpp>

#include <omp.h>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
#include <ctime>
#include <random>

const int M=128;
const int N=128;
const int P=128;
const std::string mm2D{"mm2D"};
enum schedulers{StaticSchedule, DynamicSchedule};
static const std::string scheduleNames[] = {"static", "dynamic"};

using matrix2d = Kokkos::View<int **, Kokkos::OpenMP::memory_space>;
namespace KTE = Kokkos::Tools::Experimental;
namespace KE = Kokkos::Experimental;
constexpr int lowerBound{100};
constexpr int upperBound{999};

// Helper function to generate tile sizes
std::vector<int64_t> factorsOf(const int &size){
    std::vector<int64_t> factors;
    for(int i=1; i<size; i++){
        if(size % i == 0){
            factors.push_back(i);
        }
    }
    return factors;
}

// Helper function to generate thread counts
std::vector<int64_t> makeRange(const int &size){
    std::vector<int64_t> range;
    for(int i=2; i<=size; i+=2){
        range.push_back(i);
    }
    return range;
}

// helper function for human output
void reportOptions(std::vector<int64_t>& candidates,
    std::string name, size_t size) {
    std::cout<<"Tiling options for " << name << "="<<size<<std::endl;
    for(auto &i : candidates){ std::cout<<i<<", "; }
    std::cout<<std::endl;
}

// helper function for matrix init
void initArray(matrix2d& ar, size_t d1, size_t d2) {
    for(size_t i=0; i<d1; i++){
        for(size_t j=0; j<d2; j++){
                ar(i,j)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
        }
    }
}

// helper function for matrix init
void zeroArray(matrix2d& ar, size_t d1, size_t d2) {
    for(size_t i=0; i<d1; i++){
        for(size_t j=0; j<d2; j++){
                ar(i,j)=0.0;
        }
    }
}

// helper function for declaring output tiling variables
size_t declareOutputTileSize(std::string name, std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = factorsOf(limit);
    reportOptions(candidates, name, limit);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring input size variables
size_t declareInputViewSize(std::string varname, int64_t size) {
    size_t in_value_id;
    // create a 'vector' of value(s)
    std::vector<int64_t> candidates = {size};
    // create our variable object
    KTE::VariableInfo in_info;
    // set the variable details
    in_info.type = KTE::ValueType::kokkos_value_int64;
    in_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    in_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    in_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    in_value_id = KTE::declare_input_type(varname,in_info);
    // return the id
    return in_value_id;
}

// helper function for declaring scheduler variable
size_t declareOutputSchedules(std::string varname) {
    // create a vector of potential values
    std::vector<int64_t> candidates_schedule = {StaticSchedule,DynamicSchedule};
    // create our variable object
    KTE::VariableInfo schedule_out_info;
    // set the variable details
    schedule_out_info.type = KTE::ValueType::kokkos_value_int64;
    schedule_out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    schedule_out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    schedule_out_info.candidates = KTE::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());
    // declare the variable
    size_t schedule_out_value_id = KTE::declare_output_type(varname,schedule_out_info);
    // return the id
    return schedule_out_value_id;
}

// helper function for declaring output tread count variable
size_t declareOutputThreadCount(std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = makeRange(limit);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

int main(int argc, char *argv[]){
    // surely there is a way to get this from Kokkos?
    bool tuning = false;
    char * tmp{getenv("APEX_KOKKOS_TUNING")};
    if (tmp != nullptr) {
        std::string tmpstr {tmp};
        if (tmpstr.compare("1") == 0) {
            tuning = true;
        }
    }
    // initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // print the Kokkos configuration
        Kokkos::print_configuration(std::cout, false);
        // seed the random number generator, sure
        srand(time(0));

        /* Declare/Init re,ar1,ar2 */
        matrix2d ar1("array1",M,N), ar2("array2",N,P), re("Result",M,P);
        initArray(ar1, M, N);
        initArray(ar2, N, P);
        zeroArray(re, M, P);

        // Context variable setup - needed to generate a unique context hash for tuning.
        // Declare the variables and store the variable IDs
        size_t id[5];
        id[0] = 1; // default input for the region name ("mm2D")
        id[1] = 2; // default input for the region type ("parallel_for")
        id[2] = declareInputViewSize("matrix_size_M", M);
        id[3] = declareInputViewSize("matrix_size_N", N);
        id[4] = declareInputViewSize("matrix_size_P", P);

        // create an input vector of variables with name, loop type, and view sizes.
        std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(id[0], mm2D),
            KTE::make_variable_value(id[1], "parallel_for"),
            KTE::make_variable_value(id[2], int64_t(M)),
            KTE::make_variable_value(id[3], int64_t(N)),
            KTE::make_variable_value(id[4], int64_t(P))
        };

        // Declare the variables and store the variable IDs
        size_t out_value_id[5];

        // Tuning tile size - setup
        out_value_id[0] = declareOutputTileSize("M", "ti_out", M);
        out_value_id[1] = declareOutputTileSize("N", "tj_out", N);
        out_value_id[2] = declareOutputTileSize("P", "tk_out", P);
        // Tuning tile size - end setup

        // scheduling policy - setup
        out_value_id[3] = declareOutputSchedules("schedule_out");
        // scheduling policy - end setup

        // thread count - setup
        int64_t max_threads = std::min(std::thread::hardware_concurrency(),
                (unsigned int)(Kokkos::OpenMP::concurrency()));
        out_value_id[4] = declareOutputThreadCount("thread_count", max_threads);
        // thread count - end setup

        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_value_id[0], int64_t(1)),
            KTE::make_variable_value(out_value_id[1], int64_t(1)),
            KTE::make_variable_value(out_value_id[2], int64_t(1)),
            KTE::make_variable_value(out_value_id[3], int64_t(StaticSchedule)),
            KTE::make_variable_value(out_value_id[4], int64_t(max_threads))
        };

        /* Declare the kernel that does the work */
        const auto kernel = KOKKOS_LAMBDA(int i, int j, int k){
            re(i,j) += ar1(i,j) * ar2(j,k);
        };

        /* Iterate max_iterations times, so that we can explore the search
         * space. Not all searches will converge - we have a large space!
         * It's likely that exhaustive search will fail to converge. */
        for (int i = 0 ; i < Impl::max_iterations ; i++) {
            // request a context id
            size_t context = KTE::get_new_context_id();
            // start the context
            KTE::begin_context(context);

            // set the input values for the context
            KTE::set_input_values(context, input_vector.size(), input_vector.data());
            // request new output values for the context
            KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

            // get the tiling factors
            int ti,tj,tk;
            ti = answer_vector[0].value.int_value;
            tj = answer_vector[1].value.int_value;
            tk = answer_vector[2].value.int_value;
            // get our schedule and thread count
            int scheduleType = answer_vector[3].value.int_value;
            // there's probably a better way to set the thread count?
            int num_threads = answer_vector[4].value.int_value;
            int leftover_threads = max_threads - answer_vector[4].value.int_value;

            // no tuning?
            if (!tuning) {
                // Report the tuning, if desired
                /*
                std::cout << "Tiling: default, ";
                std::cout << "Schedule: default ";
                std::cout << "Threads: default ";
                std::cout << std::endl;
                */

                // default scheduling policy, default tiling
                Kokkos::MDRangePolicy<Kokkos::OpenMP,
                    Kokkos::Rank<3>> default_policy({0,0,0},{M,N,P});
                Kokkos::parallel_for(
                        mm2D, default_policy, KOKKOS_LAMBDA(int i, int j, int k){
                        re(i,j) += ar1(i,j) * ar2(j,k);
                        }
                        );
            // use static schedule?
            } else if (scheduleType == StaticSchedule) {
                // Report the tuning, if desired
                /*
                std::cout << "Tiling: [" << ti << "," << tj << "," << tk << "], ";
                std::cout << "Schedule: " << scheduleNames[scheduleType] << ", ";
                std::cout << "Threads: " << answer_vector[4].value.int_value;
                std::cout << std::endl;
                */

                // if using max threads, no need to partition
                if (num_threads == max_threads) {
                    // static scheduling, tuned tiling
                    Kokkos::MDRangePolicy<Kokkos::OpenMP,
                        Kokkos::Schedule<Kokkos::Static>,
                        Kokkos::Rank<3>> static_policy({0,0,0},{M,N,P},{ti,tj,tk});
                    Kokkos::parallel_for(mm2D, static_policy, kernel);
                } else {
                    // partition the space so we can tune the number of threads
                    auto instances = KE::partition_space(Kokkos::OpenMP(),
                        num_threads, leftover_threads);
                    // static scheduling, tuned tiling
                    Kokkos::MDRangePolicy<Kokkos::OpenMP,
                        Kokkos::Schedule<Kokkos::Static>,
                        Kokkos::Rank<3>> static_policy(instances[0],{0,0,0},{M,N,P},{ti,tj,tk});
                    Kokkos::parallel_for(mm2D, static_policy, kernel);
                }
            } else {
                /*
                // Report the tuning, if desired
                std::cout << "Tiling: [" << ti << "," << tj << "," << tk << "], ";
                std::cout << "Schedule: " << scheduleNames[scheduleType] << ", ";
                std::cout << "Threads: " << answer_vector[4].value.int_value;
                std::cout << std::endl;
                */

                // if using max threads, no need to partition
                if (num_threads == max_threads) {
                    // dynamic scheduling, tuned tiling
                    Kokkos::MDRangePolicy<Kokkos::OpenMP,
                        Kokkos::Schedule<Kokkos::Dynamic>,
                        Kokkos::Rank<3>> dynamic_policy({0,0,0},{M,N,P},{ti,tj,tk});
                    Kokkos::parallel_for(
                            mm2D, dynamic_policy, kernel);
                } else {
                    // partition the space so we can tune the number of threads
                    auto instances = KE::partition_space(Kokkos::OpenMP(),
                        num_threads, leftover_threads);
                    // dynamic scheduling, tuned tiling
                    Kokkos::MDRangePolicy<Kokkos::OpenMP,
                        Kokkos::Schedule<Kokkos::Dynamic>,
                        Kokkos::Rank<3>> dynamic_policy(instances[0],{0,0,0},{M,N,P},{ti,tj,tk});
                    Kokkos::parallel_for(
                            mm2D, dynamic_policy, kernel);
                }
            }
            // end the context
            KTE::end_context(context);
        }
    }
    Kokkos::finalize();
}