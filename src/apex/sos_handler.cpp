//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "sos_handler.hpp"
#include "apex.hpp"
#include "apex_api.hpp"

using namespace std;
using namespace apex;

namespace apex {

sos_handler::sos_handler (int argc, char * argv[], int period) : handler(period), _terminate(false), _runtime(NULL), _pub(NULL) {
    timer.c_val = strdup("Timer");
    counter.c_val = strdup("Counter");
    _runtime = SOS_init(&argc, &argv, SOS_ROLE_CLIENT, SOS_LAYER_LIB);
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

bool sos_handler::_handler(void) {
    static int old_count = 0;
    int new_count = 0;
    if (_pub == NULL) {
        _make_pub();
    }
    assert(_pub);
    if (_terminate) { return false; }
    std::unordered_map<task_identifier, profile*> task_map = 
        apex::instance()->the_profiler_listener->get_task_map();
    std::cout << "Iterating..." << std::endl;
    for (const auto iter : task_map) {
        new_count++;
        task_identifier task = iter.first;
        profile* p = iter.second;
        std::cout << task.get_name() << std::endl;
        SOS_val calls, mean, accumulated, minimum, maximum, variance, sum_squares, stddev;
        calls.i_val = p->get_calls();
        SOS_pack(_pub, "Number of Calls", SOS_VAL_TYPE_INT, calls);
        accumulated.d_val = p->get_accumulated();
        SOS_pack(_pub, "Accumulated", SOS_VAL_TYPE_DOUBLE, accumulated);
        mean.d_val = p->get_mean();
        SOS_pack(_pub, "Mean", SOS_VAL_TYPE_DOUBLE, mean);
        if (p->get_type() == APEX_TIMER) {
            SOS_pack(_pub, "Type", SOS_VAL_TYPE_STRING, timer);
        } else {
            SOS_pack(_pub, "Type", SOS_VAL_TYPE_STRING, counter);
            minimum.d_val = p->get_minimum();
            SOS_pack(_pub, "Minimum", SOS_VAL_TYPE_DOUBLE, minimum);
            maximum.d_val = p->get_maximum();
            SOS_pack(_pub, "Maximum", SOS_VAL_TYPE_DOUBLE, maximum);
            variance.d_val = p->get_variance();
            SOS_pack(_pub, "Variance", SOS_VAL_TYPE_DOUBLE, variance);
            sum_squares.d_val = p->get_sum_squares();
            SOS_pack(_pub, "Sum of Squares", SOS_VAL_TYPE_DOUBLE, sum_squares);
            stddev.d_val = p->get_stddev();
            SOS_pack(_pub, "Variance", SOS_VAL_TYPE_DOUBLE, stddev);
        }
    }
    if (new_count > SOS_DEFAULT_ELEM_MAX) {
        std::cerr << "DANGER, WILL ROBINSON! EXCEEDING MAX ELEMENTS IN SOS. Bad things might happen?\n";
    }
    if (new_count > old_count) {
        SOS_announce(_pub);
    }
    SOS_publish(_pub);
    return true;
}


}
