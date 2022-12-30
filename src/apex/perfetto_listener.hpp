/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "perfetto_static.hpp"
#include "event_listener.hpp"
#include "async_thread_node.hpp"

namespace apex {

class perfetto_listener : public event_listener {

public:
  	perfetto_listener (void);
  	~perfetto_listener (void);
  	void on_startup(startup_event_data &data);
  	void on_dump(dump_event_data &data);
  	void on_reset(task_identifier * id) { APEX_UNUSED(id); };
  	void on_pre_shutdown(void);
  	void on_shutdown(shutdown_event_data &data);
  	void on_new_node(node_event_data &data);
  	void on_new_thread(new_thread_event_data &data);
  	void on_exit_thread(event_data &data);
  	bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
  	void on_stop(std::shared_ptr<profiler> &p);
  	void on_yield(std::shared_ptr<profiler> &p);
  	bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
  	void _common_stop(std::shared_ptr<profiler> &p);
  	bool _common_start(std::shared_ptr<task_wrapper> &tt_ptr);
  	void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
    	APEX_UNUSED(tt_ptr);
  	};
  	void on_sample_value(sample_value_event_data &data);
  	void on_periodic(periodic_event_data &data);
  	void on_custom_event(custom_event_data &data);
  	void on_send(message_event_data &data) { APEX_UNUSED(data); };
  	void on_recv(message_event_data &data) { APEX_UNUSED(data); };
  	void set_node_id(int node_id, int node_count);
    void on_async_event(base_thread_node &node, std::shared_ptr<profiler> &p,
        const async_event_data& data);
    void on_async_metric(base_thread_node &node, std::shared_ptr<profiler> &p);

private:
    void get_file_name();
    size_t make_tid (base_thread_node &node);
    std::unique_ptr<perfetto::TracingSession> tracing_session;
    std::string filename;
    int file_descriptor;
    std::mutex _vthread_mutex;
    std::map<base_thread_node, size_t> vthread_map;

};

}

