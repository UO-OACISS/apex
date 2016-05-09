//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "apex.hpp"
#include "apex_api.hpp"
#include "sos_handler.hpp"
#include <cmath>
#include <ctgmath>

using namespace std;
using namespace apex;

namespace apex {

sos_handler::sos_handler (int argc, char * argv[], int period) : handler(period), _terminate(false), _runtime(NULL), _pub(NULL) {
    timer = strdup("Timer");
    counter = strdup("Counter");
    _runtime = SOS_init(&argc, &argv, SOS_ROLE_CLIENT, SOS_LAYER_LIB);
    // it will be.
    if (_pub == NULL) {
        _make_pub();
    }
    run();
}

sos_handler::~sos_handler() {
    SOS_finalize(_runtime);
}

void sos_handler::_make_pub(void) {
    char pub_name[SOS_DEFAULT_STRING_LEN] = {0};
    char app_version[SOS_DEFAULT_STRING_LEN] = {0};

    printf("[_make_pub]: Creating new pub...\n");
#ifdef APEX_MPI
    int rank;
    int commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    _runtime->config.comm_rank = rank;
    _runtime->config.comm_size = commsize;
#else
    _runtime->config.comm_rank = apex::instance()->get_node_id();
    _runtime->config.comm_size = 1;
#endif
    sprintf(pub_name, "APEX_SOS_SUPPORT");
    sprintf(app_version, "%s", version().c_str());
    _pub = SOS_pub_create(_runtime, pub_name, SOS_NATURE_DEFAULT);
    strcpy(_pub->prog_ver, app_version);
    _pub->meta.channel       = 1;
    _pub->meta.layer         = SOS_LAYER_LIB;
    printf("[_make_pub]:   ... done.  (pub->guid == %ld)\n", _pub->guid);
    return;
}

bool sos_handler::_handler_internal(void) {
    static int old_count = 0;
    int new_count = 0;
    if (_pub == NULL) {
        _make_pub();
    }
    assert(_pub);
    std::unordered_map<task_identifier, profile*> task_map = 
        apex::instance()->the_profiler_listener->get_task_map();
    for (const auto iter : task_map) {
        task_identifier task = iter.first;
        profile* p = iter.second;
        int calls; 
        double mean, accumulated, minimum, maximum, variance, sum_squares, stddev;
        calls = p->get_calls();
        SOS_pack(_pub, ("APEX::" + task.get_name() + "::Number of Calls").c_str(), SOS_VAL_TYPE_INT, &calls);
        new_count++;
        accumulated = p->get_accumulated();
        if (isinf(accumulated)) { accumulated = 0.0; }
        SOS_pack(_pub, ("APEX::" + task.get_name() + "::Accumulated").c_str(), SOS_VAL_TYPE_DOUBLE, &accumulated);
        new_count++;
        mean = p->get_mean();
        if (isinf(mean)) { mean = 0.0; }
        SOS_pack(_pub, ("APEX::" + task.get_name() + "::Mean").c_str(), SOS_VAL_TYPE_DOUBLE, &mean);
        new_count++;
        if (p->get_type() == APEX_TIMER) {
            //SOS_pack(_pub, ("APEX::" + task.get_name() + "::Type").c_str(), SOS_VAL_TYPE_STRING, timer);
        } else {
            //SOS_pack(_pub, ("APEX::" + task.get_name() + "::Type").c_str(), SOS_VAL_TYPE_STRING, counter);
            minimum = p->get_minimum();
            if (isinf(minimum)) { minimum = 0.0; }
            SOS_pack(_pub, ("APEX::" + task.get_name() + "::Minimum").c_str(), SOS_VAL_TYPE_DOUBLE, &minimum);
        new_count++;
            maximum = p->get_maximum();
            if (isinf(maximum)) { maximum = 0.0; }
            SOS_pack(_pub, ("APEX::" + task.get_name() + "::Maximum").c_str(), SOS_VAL_TYPE_DOUBLE, &maximum);
        new_count++;
            variance = p->get_variance();
            if (isinf(variance)) { variance = 0.0; }
            SOS_pack(_pub, ("APEX::" + task.get_name() + "::Variance").c_str(), SOS_VAL_TYPE_DOUBLE, &variance);
        new_count++;
            sum_squares = p->get_sum_squares();
            if (isinf(sum_squares)) { sum_squares = 0.0; }
            SOS_pack(_pub, ("APEX::" + task.get_name() + "::Sum of Squares").c_str(), SOS_VAL_TYPE_DOUBLE, &sum_squares);
        new_count++;
            stddev = p->get_stddev();
            if (isinf(stddev)) { stddev = 0.0; }
            SOS_pack(_pub, ("APEX::" + task.get_name() + "::Stddev").c_str(), SOS_VAL_TYPE_DOUBLE, &stddev);
        new_count++;
        }
    }
    if (new_count > SOS_DEFAULT_ELEM_MAX) {
        std::cerr << "DANGER, WILL ROBINSON! EXCEEDING MAX ELEMENTS IN SOS. Bad things might happen?\n";
    }
    if (new_count > old_count) {
        SOS_announce(_pub);
    }
    old_count = new_count;
    SOS_publish(_pub);
    return true;
}

bool sos_handler::_handler(void) {
	std::unique_lock<std::mutex> l(_terminate_mutex);
    if (_terminate) { 
        return false; 
    } else {
        _handler_internal();
    }
}

void sos_handler::terminate(void) {
    {
	    std::unique_lock<std::mutex> l(_terminate_mutex);
	    _terminate = true;
    }
    _handler_internal();
}


}
