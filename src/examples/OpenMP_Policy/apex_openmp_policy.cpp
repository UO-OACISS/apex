//  APEX OpenMP Policy
//
//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <set>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <stdio.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <omp.h>

#include "apex_api.hpp"
#include "apex_policies.hpp"


static int apex_openmp_policy_tuning_window = 3;
static apex_ah_tuning_strategy apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
static std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>> * apex_openmp_policy_tuning_requests;
static bool apex_openmp_policy_verbose = false;
static bool apex_openmp_policy_use_history = false;
static bool apex_openmp_policy_running = false;
static std::string apex_openmp_policy_history_file = "";

static const std::list<std::string> default_thread_space{"2", "4", "8", "16", "24", "32"};
static const std::list<std::string> default_schedule_space{"static", "dynamic", "guided"};
static const std::list<std::string> default_chunk_space{"1", "8", "32", "64", "128", "256", "512"};
static uint32_t thread_cap = 1; // it'll grow from here

static const std::list<std::string> * thread_space = nullptr;
static const std::list<std::string> * schedule_space = nullptr;
static const std::list<std::string> * chunk_space = nullptr;

static apex_policy_handle * start_policy_handle;
static apex_policy_handle * stop_policy_handle;

static void set_omp_params(std::shared_ptr<apex_tuning_request> request) {
	//std::cout << __func__ << std::endl;
        std::shared_ptr<apex_param_enum> thread_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_num_threads"));
        const int num_threads = atoi(thread_param->get_value().c_str());

        std::shared_ptr<apex_param_enum> schedule_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_schedule"));
        const std::string schedule_value = schedule_param->get_value();
        omp_sched_t schedule = omp_sched_auto;
        if(schedule_value == "static") {
            schedule = omp_sched_static;
        } else if(schedule_value == "dynamic") {
            schedule = omp_sched_dynamic;
        } else if(schedule_value == "guided") {
            schedule = omp_sched_guided;
        } else if(schedule_value == "auto") {
            schedule = omp_sched_auto;
        } else {
            throw std::invalid_argument("omp_schedule");
        }

        std::shared_ptr<apex_param_enum> chunk_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_chunk_size"));
        const int chunk_size = atoi(chunk_param->get_value().c_str());

        const char * name = request->get_name().c_str();

        if(apex_openmp_policy_verbose) {
            fprintf(stderr, "name: %s, num_threads: %d, schedule %d, chunk_size %d\n", name, num_threads, schedule, chunk_size);
        }

        omp_set_num_threads(num_threads);
        omp_set_schedule(schedule, chunk_size);
}


void handle_start(const std::string & name) {
	//std::cout << __func__ << std::endl;
    auto search = apex_openmp_policy_tuning_requests->find(name);
    if(search == apex_openmp_policy_tuning_requests->end()) {
        // Start a new tuning session.
        if(apex_openmp_policy_verbose) {
            fprintf(stderr, "Starting tuning session for %s\n", name.c_str());
        }
        std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
        apex_openmp_policy_tuning_requests->insert(std::make_pair(name, request));

        // Create an event to trigger this tuning session.
        apex_event_type trigger = apex::register_custom_event(name);
        request->set_trigger(trigger);

        // Create a metric
        std::function<double(void)> metric = [=]()->double{
            apex_profile * profile = apex::get_profile(name);
            if(profile == nullptr) {
                std::cerr << "ERROR: no profile for " << name << std::endl;
                return 0.0;
            }
            if(profile->calls == 0.0) {
                std::cerr << "ERROR: calls = 0 for " << name << std::endl;
                return 0.0;
            }
            double result = profile->accumulated/profile->calls;
            if(apex_openmp_policy_verbose) {
                fprintf(stderr, "time per call: %f\n", result);
            }
            return result;
        };
        request->set_metric(metric);

        // Set apex_openmp_policy_tuning_strategy
        request->set_strategy(apex_openmp_policy_tuning_strategy);

        //int max_threads = omp_get_num_procs();

        // Create a parameter for number of threads.
        std::shared_ptr<apex_param_enum> threads_param = request->add_param_enum("omp_num_threads", std::to_string(thread_cap), *thread_space);

        // Create a parameter for scheduling policy.
        std::shared_ptr<apex_param_enum> schedule_param = request->add_param_enum("omp_schedule", "static", *schedule_space);

        // Create a parameter for chunk size.
        std::shared_ptr<apex_param_enum> chunk_param = request->add_param_enum("omp_chunk_size", "64", *chunk_space);

        // Set OpenMP runtime parameters to initial values.
        set_omp_params(request);

        // Start the tuning session.
        //apex_tuning_session_handle session = apex::setup_custom_tuning(*request);
        apex::setup_custom_tuning(*request);
    } else {
        // We've seen this region before.
        std::shared_ptr<apex_tuning_request> request = search->second;
        set_omp_params(request);
    }
}

void handle_stop(const std::string & name) {
	//std::cout << __func__ << std::endl;
    auto search = apex_openmp_policy_tuning_requests->find(name);
    if(search == apex_openmp_policy_tuning_requests->end()) {
        std::cerr << "ERROR: Stop received on \"" << name << "\" but we've never seen a start for it." << std::endl;
    } else {
        apex_profile * profile = apex::get_profile(name);
        if(apex_openmp_policy_tuning_window == 1 || (profile != nullptr && profile->calls >= apex_openmp_policy_tuning_window)) {
            //std::cout << "Num calls: " << profile->calls << std::endl;
            std::shared_ptr<apex_tuning_request> request = search->second;
            // Evaluate the results
            apex::custom_event(request->get_trigger(), NULL);
            // Reset counter so each measurement is fresh.
            apex::reset(name);
        }
    }
}

int start_policy(const apex_context context) {
	//std::cout << __func__ << std::endl;
    if(context.data == nullptr) {
        std::cerr << "ERROR: No task_identifier for event!" << std::endl;
        return APEX_ERROR;
    }
    apex::task_identifier * id = (apex::task_identifier *) context.data;
    if(!id->has_name) {
        // Skip events without names.
        return APEX_NOERROR;
    }
    std::string name = id->get_name(false);
    if(name.find("OpenMP Parallel Region:") == 0) {
        handle_start(name);
    }
    return APEX_NOERROR;
}

int stop_policy(const apex_context context) {
	//std::cout << __func__ << std::endl;
    if(context.data == nullptr) {
        std::cerr << "ERROR: No task_identifier for event!" << std::endl;
        return APEX_ERROR;
    }
    apex::task_identifier * id = (apex::task_identifier *) context.data;
    if(!id->has_name) {
        // Skip events without names.
        return APEX_NOERROR;
    }
    std::string name = id->get_name(false);
    if(name.find("OpenMP Parallel Region:") == 0) {
        handle_stop(name);
    }
    return APEX_NOERROR;
}

void Tokenize(const std::string& str,
                      std::vector<std::string>& tokens,
                      const std::string& delimiters = "$")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

void read_results(const std::string & filename) {
    std::ifstream results_file(filename, std::ifstream::in);
    if(!results_file.good()) {
        std::cerr << "Unable to open results file " << filename << std::endl;
        assert(false);
    } else {
        std::string line;
        std::getline(results_file, line); // ignore first line (header)
        while(!results_file.eof()) {
            std::getline(results_file, line);
            std::vector<std::string> parts;
            Tokenize(line, parts);
            if(parts.size() == 5) {
                std::string & name = parts[0];
                std::string & threads = parts[1];
                std::string & schedule = parts[2];
                std::string & chunk_size = parts[3];
                std::string & converged = parts[4];
                // Remove quotes from strings
                name.erase(std::remove(name.begin(), name.end(), '"'), name.end());
                threads.erase(std::remove(threads.begin(), threads.end(), '"'), threads.end());
                schedule.erase(std::remove(schedule.begin(), schedule.end(), '"'), schedule.end());
                chunk_size.erase(std::remove(chunk_size.begin(), chunk_size.end(), '"'), chunk_size.end());
                converged.erase(std::remove(converged.begin(), converged.end(), '"'), converged.end());
                // Create a dummy tuning request with the values from the results file.
                std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
                apex_openmp_policy_tuning_requests->insert(std::make_pair(name, request));
                std::shared_ptr<apex_param_enum> threads_param = request->add_param_enum("omp_num_threads", threads, {threads});
                std::shared_ptr<apex_param_enum> schedule_param = request->add_param_enum("omp_schedule", schedule, {schedule});
                std::shared_ptr<apex_param_enum> chunk_param = request->add_param_enum("omp_chunk_size", chunk_size, {chunk_size});

                if(apex_openmp_policy_verbose) {
                   fprintf(stderr, "Added %s -> (%s, %s, %s) from history.\n", name.c_str(), threads.c_str(), schedule.c_str(), chunk_size.c_str());
                }
            }
        }
    }
}

void print_summary() {
    std::time_t time = std::time(NULL);
    char time_str[128];
    std::strftime(time_str, 128, "results-%F-%H-%M-%S.csv", std::localtime(&time));
    std::ofstream results_file(time_str, std::ofstream::out);
    results_file << "\"name\",\"num_threads\",\"schedule\",\"chunk_size\",\"converged\"" << std::endl;
    if(apex_openmp_policy_verbose) {
    	std::cout << std::endl << "OpenMP final settings: " << std::endl;
	}
    for(auto request_pair : *apex_openmp_policy_tuning_requests) {
        auto request = request_pair.second;
        request->get_best_values();
        const std::string & name = request->get_name();
        const std::string & threads = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_num_threads"))->get_value();
        const std::string & schedule = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_schedule"))->get_value();
        const std::string & chunk = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_chunk_size"))->get_value();
        const std::string converged = request->has_converged() ? "CONVERGED" : "NOT CONVERGED";
        if(apex_openmp_policy_verbose) {
        	std::cout << "name: " << name << ", num_threads: " << threads << ", schedule: " << schedule
            	<< ", chunk_size: " << chunk << " " << converged << std::endl;
		}
        results_file << "\"" << name << "\"$" << threads << "$\"" << schedule << "\"$" << chunk << "$\"" << converged << "\"" << std::endl;
    }
    std::cout << std::endl;
    results_file.flush();
    results_file.close();
}

bool parse_space_file(const std::string & filename) {
    using namespace rapidjson;
    std::ifstream space_file(filename, std::ifstream::in);
    if(!space_file.good()) {
        std::cerr << "Unable to open parameter space specification file " << filename << std::endl;
        assert(false);
        return false;
    } else {
        IStreamWrapper space_file_wrapper(space_file);
        Document document;
        document.ParseStream(space_file_wrapper);
        if(!document.IsObject()) {
            std::cerr << "Parameter space file root must be an object." << std::endl;
            return false;
        }
        if(!document.HasMember("tuning_space")) {
            std::cerr << "Parameter space file root must contain a member named 'tuning_space'." << std::endl;
            return false;
        }

        const auto & tuning_spec = document["tuning_space"];
        if(!tuning_spec.IsObject()) {
            std::cerr << "Parameter space file's 'tuning_space' member must be an object." << std::endl;
            return false;
        }
        if(!tuning_spec.HasMember("omp_num_threads")) {
            std::cerr << "Parameter space file's 'tuning_space' object must contain a member named 'omp_num_threads'" << std::endl;
            return false;
        }
        if(!tuning_spec.HasMember("omp_schedule")) {
            std::cerr << "Parameter space file's 'tuning_space' object must contain a member named 'omp_schedule'" << std::endl;
            return false;
        }
        if(!tuning_spec.HasMember("omp_chunk_size")) {
            std::cerr << "Parameter space file's 'tuning_space' object must contain a member named 'omp_chunk_size'" << std::endl;
            return false;
        }

        const auto & omp_num_threads_array = tuning_spec["omp_num_threads"];
        const auto & omp_schedule_array    = tuning_spec["omp_schedule"];
        const auto & omp_chunk_size_array  = tuning_spec["omp_chunk_size"];

        // Validate array types
        if(!omp_num_threads_array.IsArray()) {
            std::cerr << "Parameter space file's 'omp_num_threads' member must be an array." << std::endl;
            return false;
        }
        if(!omp_schedule_array.IsArray()) {
            std::cerr << "Parameter space file's 'omp_schedule' member must be an array." << std::endl;
            return false;
        }
        if(!omp_chunk_size_array.IsArray()) {
            std::cerr << "Parameter space file's 'omp_chunk_size' member must be an array." << std::endl;
            return false;
        }

        // omp_num_threads
        std::list<std::string> num_threads_list;
        for(auto itr = omp_num_threads_array.Begin(); itr != omp_num_threads_array.End(); ++itr) {
              if(itr->IsInt()) {
                  const unsigned int this_num_threads = itr->GetInt();
                  if (this_num_threads <= apex::hardware_concurrency()) {
                      const std::string this_num_threads_str = std::to_string(this_num_threads);
                      num_threads_list.push_back(this_num_threads_str);
                      thread_cap = thread_cap < this_num_threads ? this_num_threads : thread_cap;
                  }
              } else if(itr->IsString()) {
                  const char * this_num_threads = itr->GetString();
                  unsigned int tmpint = (unsigned int)(atoi(this_num_threads));
                  if (tmpint <= apex::hardware_concurrency()) {
                      const std::string this_num_threads_str = std::string(this_num_threads, itr->GetStringLength());
                      num_threads_list.push_back(this_num_threads_str);
                      thread_cap = thread_cap < tmpint ? tmpint : thread_cap;
                  }
              } else {
                  std::cerr << "Parameter space file's 'omp_num_threads' member must contain only integers or strings" << std::endl;
                  return false;
              }
        }
        thread_space = new std::list<std::string>{num_threads_list};

        // omp_schedule
        std::list<std::string> schedule_list;
        for(auto itr = omp_schedule_array.Begin(); itr != omp_schedule_array.End(); ++itr) {
              if(itr->IsString()) {
                  const char * this_schedule = itr->GetString();
                  const std::string this_schedule_str = std::string(this_schedule, itr->GetStringLength());
                  schedule_list.push_back(this_schedule_str);
              } else {
                  std::cerr << "Parameter space file's 'omp_schedule' member must contain only strings" << std::endl;
                  return false;
              }
        }
        schedule_space = new std::list<std::string>{schedule_list};

        // omp_chunk_size
        std::list<std::string> chunk_size_list;
        for(auto itr = omp_chunk_size_array.Begin(); itr != omp_chunk_size_array.End(); ++itr) {
              if(itr->IsInt()) {
                  const int this_chunk_size = itr->GetInt();
                  const std::string this_chunk_size_str = std::to_string(this_chunk_size);
                  chunk_size_list.push_back(this_chunk_size_str);
              } else if(itr->IsString()) {
                  const char * this_chunk_size = itr->GetString();
                  const std::string this_chunk_size_str = std::string(this_chunk_size, itr->GetStringLength());
                  chunk_size_list.push_back(this_chunk_size_str);
              } else {
                  std::cerr << "Parameter space file's 'omp_chunk_size' member must contain only integers or strings" << std::endl;
                  return false;
              }
        }
        chunk_space = new std::list<std::string>{chunk_size_list};

    }
    return true;
}

void print_tuning_space() {
    std::cerr << "Tuning space: " << std::endl;
    std::cerr << "\tomp_num_threads: ";
    if(thread_space == nullptr) {
        std::cerr << "NULL";
    } else {
        for(auto num_threads : *thread_space) {
            std::cerr << num_threads << " ";
        }
    }
    std::cerr << std::endl;

    std::cerr << "\tomp_schedule: ";
    if(schedule_space == nullptr) {
        std::cerr << "NULL";
    } else {
        for(auto schedule : *schedule_space) {
            std::cerr << schedule << " ";
        }
    }
    std::cerr << std::endl;

    std::cerr << "\tomp_chunk_size: ";
    if(chunk_space == nullptr) {
        std::cerr << "NULL";
    } else {
        for(auto chunk_size : *chunk_space) {
            std::cerr << chunk_size << " ";
        }
    }
    std::cerr << std::endl;
}


int register_policy() {
    // Process environment variables

    // APEX_OPENMP_VERBOSE
    const char * apex_openmp_policy_verbose_option = std::getenv("APEX_OPENMP_VERBOSE");
    if(apex_openmp_policy_verbose_option != nullptr) {
        apex_openmp_policy_verbose = 1;
    }

    // APEX_OPENMP_WINDOW
    const char * option = std::getenv("APEX_OPENMP_WINDOW");
    if(option != nullptr) {
        apex_openmp_policy_tuning_window = atoi(option);
    }
    if(apex_openmp_policy_verbose) {
        std::cerr << "apex_openmp_policy_tuning_window = " << apex_openmp_policy_tuning_window << std::endl;
    }

    // APEX_OPENMP_STRATEGY
    const char * apex_openmp_policy_tuning_strategy_option = std::getenv("APEX_OPENMP_STRATEGY");
    std::string apex_openmp_policy_tuning_strategy_str = (apex_openmp_policy_tuning_strategy_option == nullptr) ? std::string() : std::string(apex_openmp_policy_tuning_strategy_option);
    transform(apex_openmp_policy_tuning_strategy_str.begin(),
       apex_openmp_policy_tuning_strategy_str.end(),
       apex_openmp_policy_tuning_strategy_str.begin(), ::toupper);
    if(apex_openmp_policy_tuning_strategy_str.empty()) {
        // default
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
        std::cerr << "Using default tuning strategy (NELDER_MEAD)" << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "EXHAUSTIVE") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::EXHAUSTIVE;
        std::cerr << "Using EXHAUSTIVE tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "RANDOM") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::RANDOM;
        std::cerr << "Using RANDOM tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "NELDER_MEAD") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
        std::cerr << "Using NELDER_MEAD tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "PARALLEL_RANK_ORDER") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::PARALLEL_RANK_ORDER;
        std::cerr << "Using PARALLEL_RANK_ORDER tuning strategy." << std::endl;
    } else {
        std::cerr << "Invalid setting for APEX_OPENMP_STRATEGY: " << apex_openmp_policy_tuning_strategy_str << std::endl;
        std::cerr << "Will use default of NELDER_MEAD." << std::endl;
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
    }

    // APEX_OPENMP_HISTORY
    const char * apex_openmp_policy_history_file_option = std::getenv("APEX_OPENMP_HISTORY");
    if(apex_openmp_policy_history_file_option != nullptr) {
        apex_openmp_policy_history_file = std::string(apex_openmp_policy_history_file_option);
        if(!apex_openmp_policy_history_file.empty()) {
            apex_openmp_policy_use_history = true;
        }
    }
    if(apex_openmp_policy_use_history) {
        std::cerr << "Using tuning history file: " << apex_openmp_policy_history_file << std::endl;
        read_results(apex_openmp_policy_history_file);
    }

    // APEX_OPENMP_SPACE
    const char * apex_openmp_policy_space_file_option = std::getenv("APEX_OPENMP_SPACE");
    bool using_space_file = false;
    if(apex_openmp_policy_space_file_option != nullptr) {
        const std::string apex_opemp_policy_space_file{apex_openmp_policy_space_file_option};
        using_space_file = parse_space_file(apex_opemp_policy_space_file);
        if(!using_space_file) {
            std::cerr << "WARNING: Unable to use tuning space file " << apex_openmp_policy_space_file_option << ". Using default tuning space instead." << std::endl;
        }
    }

    // Set up the search spaces
    if(!using_space_file) {
        if(apex_openmp_policy_verbose) {
            std::cerr << "Using default tuning space." << std::endl;
        }
        std::list<std::string> my_thread_space{"1"};
        unsigned int nthreads = 2;
        while (nthreads <= apex::hardware_concurrency()) {
            my_thread_space.push_back(std::to_string(nthreads));
            nthreads = nthreads * 2;
        }
        thread_space   = &my_thread_space;
        schedule_space = &default_schedule_space;
        chunk_space    = &default_chunk_space;
    } else {
        if(apex_openmp_policy_verbose) {
            std::cerr << "Using tuning space from " << apex_openmp_policy_space_file_option << std::endl;
        }
    }

    if(apex_openmp_policy_verbose) {
        print_tuning_space();
    }


    // Register the policy functions with APEX
    std::function<int(apex_context const&)> start_policy_fn{start_policy};
    std::function<int(apex_context const&)> stop_policy_fn{stop_policy};
    start_policy_handle = apex::register_policy(APEX_START_EVENT, start_policy_fn);
    stop_policy_handle  = apex::register_policy(APEX_STOP_EVENT,  stop_policy_fn);
    if(start_policy_handle == nullptr || stop_policy_handle == nullptr) {
        return APEX_ERROR;
    } else {
        return APEX_NOERROR;
    }
}

extern "C" {

    int apex_plugin_init() {
		std::cout << __func__ << std::endl;
        if(!apex_openmp_policy_running) {
            fprintf(stderr, "apex_openmp_policy init\n");
            apex_openmp_policy_tuning_requests = new std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>>();
            int status =  register_policy();
            apex_openmp_policy_running = true;
            return status;
        } else {
            fprintf(stderr, "Unable to start apex_openmp_policy because it is already running.\n");
            return APEX_ERROR;
        }
    }

    int apex_plugin_finalize() {
		std::cout << __func__ << std::endl;
        if(apex_openmp_policy_running) {
            fprintf(stderr, "apex_openmp_policy finalize\n");
            //apex::deregister_policy(start_policy_handle);
            //apex::deregister_policy(stop_policy_handle);
            print_summary();
            delete apex_openmp_policy_tuning_requests;
            apex_openmp_policy_running = false;
            return APEX_NOERROR;
        } else {
            fprintf(stderr, "Unable to stop apex_openmp_policy because it is not running.\n");
            return APEX_ERROR;
        }
    }

}

