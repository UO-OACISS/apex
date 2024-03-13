/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "nvtx_listener.hpp"
#include "apex_dynamic.hpp"

using namespace std;

namespace apex {

    nvtx_listener::nvtx_listener (void) : _terminate(false) {
    }

    void nvtx_listener::on_startup(startup_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_dump(dump_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_shutdown(shutdown_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_new_node(node_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_new_thread(new_thread_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_exit_thread(event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    inline bool nvtx_listener::_common_start(std::shared_ptr<task_wrapper> &tt_ptr) {
        if (!_terminate) {
            dynamic::nvtx::push(tt_ptr->get_task_id()->get_name().c_str());
        }
        return true;
    }

    bool nvtx_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
        return _common_start(tt_ptr);
    }

    bool nvtx_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
        return _common_start(tt_ptr);
    }

    inline void nvtx_listener::_common_stop(std::shared_ptr<profiler> &p) {
        APEX_UNUSED(p);
        if (!_terminate) {
            dynamic::nvtx::pop();
        }
        return;
    }

    void nvtx_listener::on_stop(std::shared_ptr<profiler> &p) {
        return _common_stop(p);
    }

    void nvtx_listener::on_yield(std::shared_ptr<profiler> &p) {
        return _common_stop(p);
    }

    void nvtx_listener::on_sample_value(sample_value_event_data &data) {
        APEX_UNUSED(data);
        if (!_terminate) {
        }
        return;
    }

    void nvtx_listener::on_periodic(periodic_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::on_custom_event(custom_event_data &data) {
        APEX_UNUSED(data);
        return;
    }

    void nvtx_listener::set_node_id(int node_id, int node_count) {
        APEX_UNUSED(node_id);
        APEX_UNUSED(node_count);
    }

    void nvtx_listener::set_metadata(const char * name, const char * value) {
        APEX_UNUSED(name);
        APEX_UNUSED(value);
    }
}

