#include "apex.h"
#include "Kokkos_Profiling_C_Interface.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
void kokkosp_init_library(int loadseq, uint64_t version,
    uint32_t ndevinfos, struct Kokkos_Profiling_KokkosPDeviceInfo* devinfos);
void kokkosp_finalize_library();
void kokkosp_declare_output_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info);
void kokkosp_declare_input_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info);
void kokkosp_request_values(
    const size_t contextId,
    const size_t numContextVariables,
    const Kokkos_Tools_VariableValue* contextVariableValues,
    const size_t numTuningVariables,
    Kokkos_Tools_VariableValue* tuningVariableValues);
}

namespace {

/* Regression test for cache replay with the newer YAML layout:
 * - cached variables include "hash" before "name"
 * - cached output IDs differ from the runtime IDs
 * - runtime declarations arrive in a different order than the cache */
constexpr const char* kConvergedCache = R"(Input_100:
  hash: 1001
  name: cache_replay.a_input
  id: 100
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [1,2]
Input_200:
  hash: 1002
  name: cache_replay.b_input
  id: 200
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [1,2]
Output_300:
  hash: 1003
  name: cache_replay.a_output
  id: 300
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [10,20]
Output_400:
  hash: 1004
  name: cache_replay.b_output
  id: 400
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [10,20]
Context_0:
  Name: "[cache_replay.a_input:1,cache_replay.b_input:2,tree_node:cache_replay_context]"
  Strategy: "exhaustive"
  Converged: true
  Results:
    NumVars: 2
    id: 300
    value: 20
    id: 400
    value: 10
)";

struct IntSetInfo {
    explicit IntSetInfo(std::initializer_list<int64_t> values) {
        std::memset(&info, 0, sizeof(info));
        storage.assign(values.begin(), values.end());
        info.type = kokkos_value_int64;
        info.category = kokkos_value_categorical;
        info.valueQuantity = kokkos_value_set;
        info.candidates.set.size = storage.size();
        info.candidates.set.values.int_value = storage.data();
    }

    Kokkos_Tools_VariableInfo info;
    std::vector<int64_t> storage;
};

Kokkos_Tools_VariableValue make_int_variable_value(size_t id, int64_t value,
    Kokkos_Tools_VariableInfo* metadata) {
    Kokkos_Tools_VariableValue variable;
    std::memset(&variable, 0, sizeof(variable));
    variable.type_id = id;
    variable.value.int_value = value;
    variable.metadata = metadata;
    return variable;
}

std::string make_cache_path() {
    std::stringstream ss;
    ss << "/tmp/apex_cache_replay_" << getpid() << ".yaml";
    return ss.str();
}

void write_converged_cache(const std::string& filename) {
    std::ofstream results(filename);
    results << kConvergedCache;
}

int report_failure(const std::string& message, const std::string& output,
    const std::string& cache_file) {
    std::cerr << message << std::endl;
    if (!output.empty()) {
        std::cerr << "Captured output:" << std::endl;
        std::cerr << output << std::endl;
    }
    unlink(cache_file.c_str());
    return 1;
}

} // namespace

int main() {
    const std::string cache_file = make_cache_path();
    write_converged_cache(cache_file);

    kokkosp_init_library(0, KOKKOSP_INTERFACE_VERSION, 0, nullptr);
    apex_set_use_kokkos_tuning(true);
    apex_set_use_kokkos_verbose(true);
    apex_set_use_screen_output(false);
    apex_set_kokkos_tuning_cache(strdup(cache_file.c_str()));

    apex_profiler_handle profiler =
        apex_start(APEX_NAME_STRING, "cache_replay_context");

    IntSetInfo input_info({1, 2});
    IntSetInfo output_info({10, 20});

    std::ostringstream capture;
    std::streambuf* old_buffer = std::cout.rdbuf(capture.rdbuf());

    const size_t b_input_id = 11;
    const size_t a_input_id = 22;
    const size_t b_output_id = 33;
    const size_t a_output_id = 44;

    kokkosp_declare_input_type("cache_replay.b_input", b_input_id,
        input_info.info);
    kokkosp_declare_input_type("cache_replay.a_input", a_input_id,
        input_info.info);
    kokkosp_declare_output_type("cache_replay.b_output", b_output_id,
        output_info.info);
    kokkosp_declare_output_type("cache_replay.a_output", a_output_id,
        output_info.info);

    std::array<Kokkos_Tools_VariableValue, 2> inputs{
        make_int_variable_value(b_input_id, 2, &input_info.info),
        make_int_variable_value(a_input_id, 1, &input_info.info)
    };

    std::array<Kokkos_Tools_VariableValue, 2> outputs{
        make_int_variable_value(b_output_id, -1, &output_info.info),
        make_int_variable_value(a_output_id, -1, &output_info.info)
    };

    kokkosp_request_values(7, inputs.size(), inputs.data(), outputs.size(),
        outputs.data());

    std::cout.rdbuf(old_buffer);
    apex_stop(profiler);
    kokkosp_finalize_library();
    apex_finalize();
    apex_cleanup();

    const std::string output = capture.str();

    int64_t a_output = -1;
    int64_t b_output = -1;
    for (const auto& value : outputs) {
        if (value.type_id == a_output_id) {
            a_output = value.value.int_value;
        } else if (value.type_id == b_output_id) {
            b_output = value.value.int_value;
        }
    }

    if (output.find("Reading cache of Kokkos tuning results") ==
        std::string::npos) {
        return report_failure(
            "Cache replay test did not read the expected cache file.",
            output, cache_file);
    }
    if (output.find("Starting tuning session") != std::string::npos) {
        return report_failure(
            "Cache replay test unexpectedly started a fresh tuning session.",
            output, cache_file);
    }
    if (a_output != 20 || b_output != 10) {
        std::stringstream ss;
        ss << "Cache replay test returned unexpected outputs: "
           << "a_output=" << a_output << ", b_output=" << b_output;
        return report_failure(ss.str(), output, cache_file);
    }

    unlink(cache_file.c_str());
    return 0;
}
