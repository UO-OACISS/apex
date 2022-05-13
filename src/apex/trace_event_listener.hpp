/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "event_listener.hpp"
#include "async_thread_node.hpp"
#include <memory>
#include <sstream>
#ifdef APEX_HAVE_ZLIB
#include "gzstream.hpp"
#else
#include <fstream>
#endif
#include <map>
#include <atomic>
#include <map>

namespace apex {

class trace_event_listener : public event_listener {

public:
  	trace_event_listener (void);
  	~trace_event_listener (void);
  	static bool initialize_tau(int argc, char** avgv);
  	inline static bool initialized(void) { return _initialized; }
  	void on_startup(startup_event_data &data);
  	void on_dump(dump_event_data &data);
  	void on_reset(task_identifier * id)
      	{ APEX_UNUSED(id); };
  	void on_pre_shutdown(void);
  	void on_shutdown(shutdown_event_data &data);
  	void on_new_node(node_event_data &data);
  	void on_new_thread(new_thread_event_data &data);
  	void on_exit_thread(event_data &data);
  	bool on_start(std::shared_ptr<task_wrapper> &tt_ptr);
  	void on_stop(std::shared_ptr<profiler> &p);
  	void on_yield(std::shared_ptr<profiler> &p);
  	bool on_resume(std::shared_ptr<task_wrapper> &tt_ptr);
  	void on_task_complete(std::shared_ptr<task_wrapper> &tt_ptr) {
    	APEX_UNUSED(tt_ptr);
  	};
  	void on_sample_value(sample_value_event_data &data);
  	void on_periodic(periodic_event_data &data);
  	void on_custom_event(custom_event_data &data);
  	void on_send(message_event_data &data) { APEX_UNUSED(data); };
  	void on_recv(message_event_data &data) { APEX_UNUSED(data); };
  	void set_node_id(int node_id, int node_count);
  	void set_metadata(const char * name, const char * value);
    void on_async_event(base_thread_node &node, std::shared_ptr<profiler> &p,
        const async_event_data& data);
    void on_async_metric(base_thread_node &node, std::shared_ptr<profiler> &p);
    void end_trace_time(void);

private:
  	void _init(void);
  	bool _terminate;
    void flush_trace(void);
    void close_trace(void);
    void flush_trace_if_necessary(void);
  	void _common_start(std::shared_ptr<task_wrapper> &tt_ptr);
  	void _common_stop(std::shared_ptr<profiler> &p);
    std::string make_tid (base_thread_node &node);
    long unsigned int get_thread_id_metadata();
  	static bool _initialized;
    size_t get_thread_index(void);
    std::mutex * get_thread_mutex(size_t index);
    std::stringstream * get_thread_stream(size_t index);
    void write_to_trace(std::stringstream& events);
    int saved_node_id;
    std::atomic<size_t> num_events;
    std::string get_file_name();
#ifdef APEX_HAVE_ZLIB
  	io::gzofstream& get_trace_file();
#else
  	std::ofstream& get_trace_file();
#endif
  	std::stringstream trace;
    std::map<size_t, std::mutex*> mutexes;
    std::map<size_t, std::stringstream*> streams;
    std::mutex _vthread_mutex;
    std::map<base_thread_node, size_t> vthread_map;
    double _end_time;
};

int initialize_worker_thread_for_tau(void);

}

