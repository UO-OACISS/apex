/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "trace_event_listener.hpp"
#include "thread_instance.hpp"
#include "apex.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>

using namespace std;

namespace apex {

bool trace_event_listener::_initialized(false);

trace_event_listener::trace_event_listener (void) : _terminate(false),
    num_events(0), _end_time(0.0) {
    _initialized = true;
}

trace_event_listener::~trace_event_listener (void) {
    close_trace();
}

void trace_event_listener::end_trace_time(void) {
    if (_end_time == 0.0) {
        _end_time = profiler::now_us();
    }
}

void trace_event_listener::on_startup(startup_event_data &data) {
    APEX_UNUSED(data);
    saved_node_id = apex::instance()->get_node_id();
    std::stringstream ss;
    ss << fixed << "{\n";
    ss << "\"displayTimeUnit\": \"ms\",\n";
    ss << "\"traceEvents\": [\n";
    ss << "{\"name\":\"APEX MAIN\""
       << ",\"ph\":\"B\",\"pid\":"
       << saved_node_id << ",\"tid\":0,\"ts\":"
       << profiler::now_us() << "},\n";
    ss << "{\"name\":\"process_name\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"args\":{\"name\":"
       << "\"Process " << saved_node_id << "\"}},\n";
    ss << "{\"name\":\"process_sort_index\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"args\":{\"sort_index\":"
       << saved_node_id << "}},\n";
    write_to_trace(ss);
    return;
}

void trace_event_listener::on_dump(dump_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void trace_event_listener::on_pre_shutdown(void) {
    end_trace_time();
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
    write_to_trace(ss);
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
        write_to_trace(ss);
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
              << ",\"ts\":" << fixed << profiler::now_us()
              << ",\"args\":{\"value\":" << fixed << data.counter_value
              << "}},\n";
        write_to_trace(ss);
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

std::string trace_event_listener::make_tid (async_thread_node &node) {
    size_t tid;
    /* There is a potential for overlap here, but not a high potential.  The CPU and the GPU
     * would BOTH have to spawn 64k+ threads/streams for this to happen. */
    if (vthread_map.count(node) == 0) {
        size_t id = vthread_map.size()+1;
        //uint32_t id_reversed = simple_reverse(id);
        uint32_t id_shifted = id << 16;
        vthread_map.insert(std::pair<async_thread_node, size_t>(node,id_shifted));
        std::stringstream ss;
        ss << "{\"name\":\"thread_name\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_shifted
           << ",\"args\":{\"name\":";
#ifdef APEX_WITH_CUDA
        ss << "\"CUDA [" << node._device << ":" << node._context
           << ":" << std::setfill('0') << setw(5) << node._stream << "]";
#endif
#ifdef APEX_WITH_HIP
        ss << "\"HIP [" << node._device
           << ":" << std::setfill('0') << setw(5) << node._queue << "]";
#endif
#if !defined(APEX_WITH_CUDA) && !defined(APEX_WITH_HIP)
        ss << "\"GPU [" << node._device << "]";
#endif
        //ss << "" << activity_to_string(node._activity);
        ss << "\"";
        ss << "}},\n";
        /* make sure the GPU threads come after CPU threads
         * by giving them a thread sort index of max int. */
        ss << "{\"name\":\"thread_sort_index\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_shifted
           << ",\"args\":{\"sort_index\":" << UINT32_MAX << "}},\n";
        write_to_trace(ss);
    }
    tid = vthread_map[node];
    std::stringstream ss;
    ss << tid;
    std::string label{ss.str()};
    return label;
}

void trace_event_listener::on_async_event(async_thread_node &node,
    std::shared_ptr<profiler> &p) {
    if (!_terminate) {
        std::stringstream ss;
        std::string tid{make_tid(node)};
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
        write_to_trace(ss);
        flush_trace_if_necessary();
    }
}

void trace_event_listener::on_async_metric(async_thread_node &node,
    std::shared_ptr<profiler> &p) {
    if (!_terminate) {
        std::stringstream ss;
        std::string tid{make_tid(node)};
        ss << "{\"name\": \"" << p->get_task_id()->get_name()
              << "\",\"ph\":\"C\",\"pid\": " << saved_node_id
              << ",\"ts\":" << fixed << p->get_stop_us()
              << ",\"args\":{\"value\":" << fixed << p->value
              << "}},\n";
        write_to_trace(ss);
        flush_trace_if_necessary();
    }
}

size_t trace_event_listener::get_thread_index(void) {
    static size_t numthreads{0};
    _vthread_mutex.lock();
    size_t tmpval = numthreads++;
    _vthread_mutex.unlock();
    return tmpval;
}

#define SERIAL 1

std::mutex* trace_event_listener::get_thread_mutex(size_t index) {
#ifdef SERIAL
    APEX_UNUSED(index);
    return &_vthread_mutex;
#else
    std::mutex * tmp;
    // control access to the map of mutex objects
    _vthread_mutex.lock();
    if (mutexes.count(index) == 0) {
        tmp = new std::mutex();
        mutexes.insert(std::pair<size_t, std::mutex*>(index, tmp));
    } else {
        tmp = mutexes[index];
    }
    _vthread_mutex.unlock();
    return tmp;
#endif
}

std::stringstream* trace_event_listener::get_thread_stream(size_t index) {
#ifdef SERIAL
    APEX_UNUSED(index);
    return &trace;
#else
    std::stringstream * tmp;
    // control access to the map of stringstream objects
    _vthread_mutex.lock();
    if (streams.count(index) == 0) {
        tmp = new std::stringstream();
        streams.insert(std::pair<size_t, std::stringstream*>(index,tmp));
    } else {
        tmp = streams[index];
    }
    _vthread_mutex.unlock();
    return tmp;
#endif
}

void trace_event_listener::write_to_trace(std::stringstream& events) {
    static APEX_NATIVE_TLS size_t index = get_thread_index();
    static APEX_NATIVE_TLS std::mutex * mtx = get_thread_mutex(index);
    static APEX_NATIVE_TLS std::stringstream * strm = get_thread_stream(index);
    mtx->lock();
    (*strm) << events.rdbuf();
    mtx->unlock();
}

void trace_event_listener::flush_trace(void) {
    //_vthread_mutex.lock();
    //auto p = scoped_timer("APEX: Buffer Flush");
    // check if the file is open
    if (!trace_file.is_open()) {
        saved_node_id = apex::instance()->get_node_id();
        std::stringstream ss;
        ss << apex_options::output_file_path() << "/";
        ss << "trace_events." << saved_node_id << ".json";
        trace_file.open(ss.str());
    }
#ifdef SERIAL
    // flush the trace
    trace_file << trace.rdbuf() << std::flush;
    // reset the buffer
    trace.str("");
#else
    size_t count = streams.size();
    std::stringstream ss;
    for (size_t index = 0 ; index < count ; index++) {
        std::mutex * mtx = get_thread_mutex(index);
        std::stringstream * strm = get_thread_stream(index);
        mtx->lock();
        ss << strm->rdbuf();
        strm->str("");
        mtx->unlock();
    }
    // flush the trace
    trace_file << ss.rdbuf() << std::flush;
#endif
    //_vthread_mutex.unlock();
}

void trace_event_listener::flush_trace_if_necessary(void) {
    auto tmp = ++num_events;
    /* flush after every million events */
    if (tmp % 100000 == 0) {
        flush_trace();
    }
}

void trace_event_listener::close_trace(void) {
    if (trace_file.is_open()) {
        std::stringstream ss;
        ss << "{\"name\":\"APEX MAIN\""
           << ", \"ph\":\"E\",\"pid\":"
           << saved_node_id << ",\"tid\":0,\"ts\":"
           << fixed << _end_time << "}\n";
        ss << "]\n";
        ss << "}\n" << std::endl;
        write_to_trace(ss);
        flush_trace();
        //printf("Closing trace...\n"); fflush(stdout);
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

