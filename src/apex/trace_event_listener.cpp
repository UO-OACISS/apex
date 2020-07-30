//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "trace_event_listener.hpp"
#include "thread_instance.hpp"
#include "apex.hpp"
#include <iostream>
#include <fstream>
#include <memory>

using namespace std;

namespace apex {

bool trace_event_listener::_initialized(false);

trace_event_listener::trace_event_listener (void) : _terminate(false),
    num_events(0), _end_time(0.0) {
    trace << fixed << "{\n";
    trace << "\"displayTimeUnit\": \"ms\",\n";
    trace << "\"traceEvents\": [\n";
    trace << "{\"name\":\"program\""
          << ",\"ph\":\"B\",\"pid\":"
          << saved_node_id << ",\"tid\":0,\"ts\":"
          << profiler::get_time_us() << "},\n";
    trace << "{\"name\":\"process_name\""
          << ",\"ph\":\"M\",\"pid\":" << saved_node_id
          << ",\"args\":{\"name\":"
          << "\"Process " << saved_node_id << "\"}},\n";
    trace << "{\"name\":\"process_sort_index\""
          << ",\"ph\":\"M\",\"pid\":" << saved_node_id
          << ",\"args\":{\"sort_index\":"
          << saved_node_id << "}},\n";
    _initialized = true;
}

trace_event_listener::~trace_event_listener (void) {
    close_trace();
}

void trace_event_listener::end_trace_time(void) {
    if (_end_time == 0.0) {
        _end_time = profiler::get_time_us();
    }
}

void trace_event_listener::on_startup(startup_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void trace_event_listener::on_dump(dump_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void trace_event_listener::on_shutdown(shutdown_event_data &data) {
    APEX_UNUSED(data);
    if (!_terminate) {
        end_trace_time();
        flush_trace();
        close_trace();
        _terminate = true;
    }
    return;
}

void trace_event_listener::on_new_node(node_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void trace_event_listener::on_new_thread(new_thread_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void trace_event_listener::on_exit_thread(event_data &data) {
    APEX_UNUSED(data);
    return;
}

bool trace_event_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
    /*
     * Do nothing - we can do a "complete" record at stop
    */
    return true;
}

bool trace_event_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
    /*
     * Do nothing - we can do a "complete" record at stop
    */
    return true;
}

int trace_event_listener::get_thread_id_metadata() {
    int tid = thread_instance::get_id();
    std::stringstream ss;
    ss << "{\"name\":\"thread_name\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"tid\":" << tid
       << ",\"args\":{\"name\":"
       << "\"CPU Thread " << tid << "\"}},\n";
    ss << "{\"name\":\"thread_sort_index\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"tid\":" << tid
       << ",\"args\":{\"sort_index\":" << tid << "}},\n";
    _vthread_mutex.lock();
    trace << ss.rdbuf();
    _vthread_mutex.unlock();
    return tid;
}

inline void trace_event_listener::_common_stop(std::shared_ptr<profiler> &p) {
    static APEX_NATIVE_TLS int tid = get_thread_id_metadata();
    if (!_terminate) {
        std::stringstream ss;
        uint64_t pguid = 0;
        if (p->tt_ptr != nullptr && p->tt_ptr->parent != nullptr) {
            pguid = p->tt_ptr->parent->guid;
        }
        ss << "{\"name\":\"" << p->get_task_id()->get_name()
              << "\",\"ph\":\"X\",\"pid\":"
              << saved_node_id << ",\"tid\":" << tid
              << ",\"ts\":" << fixed << p->get_start_us() << ", \"dur\": "
              << p->get_stop_us() - p->get_start_us()
              << ",\"args\":{\"GUID\":" << p->guid << ",\"Parent GUID\":" << pguid << "}},\n";
        _vthread_mutex.lock();
        trace << ss.rdbuf();
        _vthread_mutex.unlock();
        flush_trace_if_necessary();
    }
    return;
}

void trace_event_listener::on_stop(std::shared_ptr<profiler> &p) {
    return _common_stop(p);
}

void trace_event_listener::on_yield(std::shared_ptr<profiler> &p) {
    return _common_stop(p);
}

void trace_event_listener::on_sample_value(sample_value_event_data &data) {
    if (!_terminate) {
        std::stringstream ss;
        ss << "{\"name\": \"" << *(data.counter_name)
              << "\",\"ph\":\"C\",\"pid\": " << saved_node_id
              << ",\"ts\":" << fixed << profiler::get_time_us()
              << ",\"args\":{\"value\":" << fixed << data.counter_value
              << "}},\n";
        _vthread_mutex.lock();
        trace << ss.rdbuf();
        _vthread_mutex.unlock();
        flush_trace_if_necessary();
    }
    return;
}

void trace_event_listener::on_periodic(periodic_event_data &data) {
    APEX_UNUSED(data);
    if (!_terminate) {
    }
    return;
}

void trace_event_listener::on_custom_event(custom_event_data &data) {
    APEX_UNUSED(data);
    if (!_terminate) {
    }
    return;
}

void trace_event_listener::set_node_id(int node_id, int node_count) {
    APEX_UNUSED(node_id);
    APEX_UNUSED(node_count);
}

void trace_event_listener::set_metadata(const char * name, const char * value) {
    APEX_UNUSED(name);
    APEX_UNUSED(value);
}

std::string trace_event_listener::make_tid (uint32_t device, uint32_t context, uint32_t stream) {
    cuda_thread_node tmp(device, context, stream);
    size_t tid;
    /* There is a potential for overlap here, but not a high potential.  The CPU and the GPU
     * would BOTH have to spawn 64k+ threads/streams for this to happen. */
    if (vthread_map.count(tmp) == 0) {
        size_t id = vthread_map.size()+1;
        uint32_t id_reversed = simple_reverse(id);
        std::cout << "New GPU thread with id: " << id << " and reversed: " << id_reversed << std::endl;
        vthread_map.insert(std::pair<cuda_thread_node, size_t>(tmp,id_reversed));
        std::stringstream ss;
        ss << "{\"name\":\"thread_name\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_reversed
           << ",\"args\":{\"name\":"
           << "\"GPU Dev:" << device;
        if (context > 0) {
            ss << " Ctx:" << context;
            if (stream > 0) {
                ss << " Str:" << stream;
            }
        }
        ss << "\"";
        ss << "}},\n";
        ss << "{\"name\":\"thread_sort_index\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_reversed
           << ",\"args\":{\"sort_index\":" << simple_reverse(1L) << "}},\n";
        _vthread_mutex.lock();
        trace << ss.rdbuf();
        _vthread_mutex.unlock();
    }
    tid = vthread_map[tmp];
    std::stringstream ss;
    ss << tid;
    std::string label{ss.str()};
    return label;
}

void trace_event_listener::on_async_event(uint32_t device, uint32_t context,
    uint32_t stream, std::shared_ptr<profiler> &p) {
    if (!_terminate) {
        std::stringstream ss;
        std::string tid{make_tid(device, context, stream)};
        uint64_t pguid = 0;
        if (p->tt_ptr != nullptr && p->tt_ptr->parent != nullptr) {
            pguid = p->tt_ptr->parent->guid;
        }
        ss << "{\"name\":\"" << p->get_task_id()->get_name()
              << "\",\"ph\":\"X\",\"pid\":"
              << saved_node_id << ",\"tid\":" << tid
              << ",\"ts\":" << fixed << p->get_start_us() << ",\"dur\":"
              << p->get_stop_us() - p->get_start_us()
              << ",\"args\":{\"GUID\":" << p->guid << ",\"Parent GUID\":" << pguid << "}},\n";
        _vthread_mutex.lock();
        trace << ss.rdbuf();
        _vthread_mutex.unlock();
        flush_trace_if_necessary();
    }
}

void trace_event_listener::flush_trace(void) {
    _vthread_mutex.lock();
    // check if the file is open
    if (!trace_file.is_open()) {
        saved_node_id = apex::instance()->get_node_id();
        std::stringstream ss;
        ss << "trace_events." << saved_node_id << ".json";
        trace_file.open(ss.str());
    }
    // flush the trace
    trace_file << trace.rdbuf() << std::flush;
    // reset the buffer
    trace.str("");
    _vthread_mutex.unlock();
}

void trace_event_listener::flush_trace_if_necessary(void) {
    auto tmp = ++num_events;
    /* flush after every million events */
    if (tmp % 1000000 == 0) {
        flush_trace();
    }
}

void trace_event_listener::close_trace(void) {
    if (trace_file.is_open()) {
        trace << "{\"name\":\"program\""
            << ", \"ph\":\"E\",\"pid\":"
            << saved_node_id << ",\"tid\":0,\"ts\":"
            << _end_time << "}\n";
        trace << "]\n";
        trace << "}\n" << std::endl;
        flush_trace();
        trace_file.close();
    }
}

/* This function is used by APEX threads so that TAU knows about them. */
int initialize_worker_thread_for_trace_event(void) {
    if (trace_event_listener::initialized())
    {
    }
    return 0;
}

}// end namespace

