//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "event_listener.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <map>
#include <atomic>

namespace apex {

class trace_event_listener : public event_listener {

    class cuda_thread_node {
    public:
        uint32_t _device;
        uint32_t _context;
        uint32_t _stream;
        cuda_thread_node(uint32_t device, uint32_t context, uint32_t stream) :
            _device(device), _context(context), _stream(stream) { }
        bool operator==(const cuda_thread_node &rhs) const {
            return (_device==rhs._device &&
                    _context==rhs._context &&
                    _stream==rhs._stream);
        }
        bool operator<(const cuda_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device && _context<rhs._context) {
                return true;
            } else if (_device == rhs._device && _context==rhs._context && _stream<rhs._stream) {
                return true;
            }
            return false;
        }
    };

public:
  	trace_event_listener (void);
  	~trace_event_listener (void);
  	static bool initialize_tau(int argc, char** avgv);
  	inline static bool initialized(void) { return _initialized; }
  	void on_startup(startup_event_data &data);
  	void on_dump(dump_event_data &data);
  	void on_reset(task_identifier * id)
      	{ APEX_UNUSED(id); };
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
    void on_async_event(uint32_t device, uint32_t context,
        uint32_t stream, std::shared_ptr<profiler> &p);
    void end_trace_time(void);

private:
  	void _init(void);
  	bool _terminate;
    void flush_trace(void);
    void close_trace(void);
    void flush_trace_if_necessary(void);
  	bool _common_start(std::shared_ptr<task_wrapper> &tt_ptr);
  	void _common_stop(std::shared_ptr<profiler> &p);
    std::string make_tid (uint32_t device, uint32_t context,
        uint32_t stream);
  	static bool _initialized;
    int saved_node_id;
    std::atomic<size_t> num_events;
  	std::ofstream trace_file;
  	std::stringstream trace;
    std::mutex _vthread_mutex;
    std::map<cuda_thread_node, size_t> vthread_map;
    double _end_time;
};

int initialize_worker_thread_for_tau(void);

}

