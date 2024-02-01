/**
 * two_var
 *
 * Complexity: medium
 *
 * Tuning problem:
 *
 * This is a two-valued tuning problem, in which you need
 * both parameters to learn the answer. There are two
 * values between 0 and 11 (inclusive).
 *
 * The penalty function here is just the distance between
 * your answer and the provided value.
 *
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
auto make_value_candidates() {
  std::vector<int64_t> candidates{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int64_t *bad_candidate_impl =
      (int64_t *)malloc(sizeof(int64_t) * candidates.size());
  memcpy(bad_candidate_impl, candidates.data(),
         sizeof(int64_t) * candidates.size());
  return Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),
                                                         bad_candidate_impl);
}
int main(int argc, char *argv[]) {
  constexpr const int data_size = 1000;
  std::vector<std::string> species = {"dog", "person"};
  //size_t context = Kokkos::Tools::Experimental::get_new_context_id();
  size_t context{1};
  tuned_kernel(
      argc, argv,
      [&](const int total_iters) {
        srand(time(NULL));
        size_t x_value_id;
        size_t y_value_id;
        size_t x_answer_id;
        size_t y_answer_id;
        Kokkos::Tools::Experimental::VariableInfo x_value_info;
        x_value_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        x_value_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_ratio;
        x_value_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        x_value_info.candidates = make_value_candidates();
        Kokkos::Tools::Experimental::VariableInfo y_value_info;
        y_value_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        y_value_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_ratio;
        y_value_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        y_value_info.candidates = make_value_candidates();
        Kokkos::Tools::Experimental::VariableInfo x_answer_info;
        x_answer_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        x_answer_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_ratio;
        x_answer_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        x_answer_info.candidates = make_value_candidates();

        Kokkos::Tools::Experimental::VariableInfo y_answer_info;
        y_answer_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        y_answer_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_categorical;
        y_answer_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        y_answer_info.candidates = make_value_candidates();
        x_value_id = Kokkos::Tools::Experimental::declare_input_type(
            "tuning_playground.x_value", x_value_info);
        y_value_id = Kokkos::Tools::Experimental::declare_input_type(
            "tuning_playground.y_value", y_value_info);
        x_answer_id = Kokkos::Tools::Experimental::declare_output_type(
            "tuning_playground.x_answer", x_answer_info);
        y_answer_id = Kokkos::Tools::Experimental::declare_output_type(
            "tuning_playground.y_answer", y_answer_info);

        return std::make_tuple(x_value_id, y_value_id, x_answer_id,
                               y_answer_id);
      },
      [&](const int iter, const int total_iters, size_t x_value_id,
          size_t y_value_id, size_t x_answer_id, size_t y_answer_id) {
        int64_t x = rand() % 12;
        int64_t y = rand() % 12;
        std::vector<Kokkos::Tools::Experimental::VariableValue> feature_vector{
            Kokkos::Tools::Experimental::make_variable_value(x_value_id, x),
            Kokkos::Tools::Experimental::make_variable_value(y_value_id, y)};
        std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
            Kokkos::Tools::Experimental::make_variable_value(x_answer_id,
                                                             int64_t(0)),
            Kokkos::Tools::Experimental::make_variable_value(y_answer_id,
                                                             int64_t(0))};
        Kokkos::Tools::Experimental::begin_context(context);
        Kokkos::Tools::Experimental::set_input_values(context, 2,
                                                      feature_vector.data());
        Kokkos::Tools::Experimental::request_output_values(
            context, 2, answer_vector.data());
        auto penalty = std::abs(answer_vector[0].value.int_value - x) +
                       std::abs(answer_vector[1].value.int_value - y);
        usleep(10 * penalty);
        Kokkos::Tools::Experimental::end_context(context);
      });
}
