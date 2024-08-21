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
#include <future>
#include <thread>

using namespace std;

namespace apex {

bool trace_event_listener::_initialized(false);

trace_event_listener::trace_event_listener (void) : _terminate(false),
    num_events(0), _end_time(0.0) {
    _initialized = true;
    // set up our swappable buffer to prevent blocking at flush time
    trace = new std::stringstream();
}

trace_event_listener::~trace_event_listener (void) {
    if (!_terminate) {
        end_trace_time();
        close_trace();
        _terminate = true;
    }
    delete trace;
}

void trace_event_listener::end_trace_time(void) {
    if (_end_time == 0.0) {
        _end_time = profiler::now_us();
    }
}

void trace_event_listener::on_startup(startup_event_data &data) {
    APEX_UNUSED(data);
    saved_node_id = apex::instance()->get_node_id();
    reversed_node_id = ((uint64_t)(simple_reverse((uint32_t)saved_node_id))) << 32;
    std::stringstream ss;
    ss.precision(3);
    ss << fixed
       << "{\"name\":\"APEX Trace Begin\""
       << ",\"ph\":\"R\",\"pid\":"
       << saved_node_id << ",\"tid\":0,\"ts\":"
       << profiler::now_us() << "},\n";
    ss << fixed << "{\"name\":\"process_name\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"args\":{\"name\":"
       << "\"Process " << saved_node_id << "\"}},\n";
    ss << "{\"name\":\"process_sort_index\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"args\":{\"sort_index\":\""
       << setw(8) << setfill('0') << saved_node_id << "\"}},\n";
    write_to_trace(ss);
    return;
}

void trace_event_listener::on_dump(dump_event_data &data) {
    APEX_UNUSED(data);
    // force a flush
    flush_trace_if_necessary(true);
    return;
}

void trace_event_listener::on_pre_shutdown(void) {
    end_trace_time();
}

void trace_event_listener::on_shutdown(shutdown_event_data &data) {
    APEX_UNUSED(data);
    if (!_terminate) {
        end_trace_time();
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

inline std::string parents_to_string(std::shared_ptr<task_wrapper> tt_ptr) {
    if (tt_ptr->parents.size() == 0) {
        return std::string("0");
    }
    if (tt_ptr->parents.size() == 1) {
        APEX_ASSERT (tt_ptr->parents[0] != nullptr);
        return std::to_string(tt_ptr->parents[0]->guid);
    }
    std::string parents{""};
    std::string delimiter{"["};
    for (auto& parent : tt_ptr->parents) {
        if (parent != nullptr) {
            parents += delimiter + std::to_string(parent->guid);
            delimiter = ",";
        }
    }
    parents += "]";
    return parents;
}

inline void trace_event_listener::_common_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    static APEX_NATIVE_TLS long unsigned int tid = get_thread_id_metadata();
    if (!_terminate) {
        std::stringstream ss;
        ss.precision(3);
        ss << fixed;
        std::string pguid = parents_to_string(tt_ptr);
        ss << "{\"name\":\"" << tt_ptr->get_task_id()->get_name()
              << "\",\"cat\":\"CPU\""
              << ",\"ph\":\"B\",\"pid\":"
              << saved_node_id << ",\"tid\":" << tid
              << ",\"ts\":" << tt_ptr->prof->get_start_us()
              << ",\"args\":{\"GUID\":" << tt_ptr->prof->guid << ",\"Parent GUID\":" << pguid << "}},\n";
/* Only write the counter at the end, it's less data! */
#if APEX_HAVE_PAPI
        int i = 0;
        for (auto metric :
            apex::instance()->the_profiler_listener->get_metric_names()) {
            double start = tt_ptr->prof->papi_start_values[i++];
            // write our counter into the event stream
            ss << fixed;
            ss << "{\"name\":\"" << metric
               << "\",\"cat\":\"CPU\""
               << ",\"ph\":\"C\",\"pid\":" << saved_node_id
               << ",\"ts\":" << tt_ptr->prof->get_start_us()
               << ",\"args\":{\"value\":" << start
               << "}},\n";
        }
#endif
        write_to_trace(ss);
    }
    flush_trace_if_necessary();
    return;
}

bool trace_event_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
    if (tt_ptr->explicit_trace_start) _common_start(tt_ptr);
    return true;
}

bool trace_event_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
    if (tt_ptr->explicit_trace_start) _common_start(tt_ptr);
    return true;
}

// we need a wrapper, because we have static tid in both _common_start and _common_stop
// and call this from both.
long unsigned int trace_event_listener::get_thread_id_metadata() {
    static APEX_NATIVE_TLS long unsigned int tid = get_thread_id_metadata_internal();
    return tid;
}

long unsigned int trace_event_listener::get_thread_id_metadata_internal() {
    int tid = thread_instance::get_id();
    saved_node_id = apex::instance()->get_node_id();
    std::stringstream ss;
    ss.precision(3);
    ss << fixed
       << "{\"name\":\"thread_name\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"tid\":" << tid
       << ",\"args\":{\"name\":"
       << "\"CPU Thread " << tid << "\"}},\n";
    ss << "{\"name\":\"thread_sort_index\""
       << ",\"ph\":\"M\",\"pid\":" << saved_node_id
       << ",\"tid\":" << tid
       << ",\"args\":{\"sort_index\":\"" << setw(5) << setfill('0') << tid << "\"}},\n";
    write_to_trace(ss);
    return tid;
}

uint64_t get_flow_id() {
    static atomic<uint64_t> flow_id{0};
    return ++flow_id;
}

void write_flow_event(std::stringstream& ss, double ts, char ph,
    std::string cat, uint64_t id, uint64_t pid, uint64_t tid, std::string parent_name,
    std::string child_name) {
    ss << "{\"ts\":" << ts
       << ",\"ph\":\"" << ph
       << "\",\"cat\":\"" << cat
       << "\",\"id\":" << id
       << ",\"pid\":" << pid
       << ",\"tid\":" << tid
       << ",\"name\":\"" << parent_name << " -> " << child_name << "\"},\n";
}

inline void trace_event_listener::_common_stop(std::shared_ptr<profiler> &p) {
    static APEX_NATIVE_TLS long unsigned int tid = get_thread_id_metadata();
    static auto main_wrapper = task_wrapper::get_apex_main_wrapper();
    // With HPX, the APEX MAIN timer will be stopped on a different thread
    // than the one that started it. So... make sure we get the right TID
    // But don't worry, the thread metadata will have been written at the
    // event start.
    //long unsigned int _tid = (p->tt_ptr->explicit_trace_start ? p->thread_id : tid);
    long unsigned int _tid = p->thread_id;
    if (!_terminate) {
        std::stringstream ss;
        ss.precision(3);
        ss << fixed;
        // if the parent tid is not the same, create a flow event BEFORE the single event
        for (auto& parent : p->tt_ptr->parents) {
            if (parent != nullptr && parent != main_wrapper
#ifndef APEX_HAVE_HPX // ...except for HPX - make the flow event regardless
            && parent->thread_id != _tid
#endif
            ) {
                //std::cout << "FLOWING!" << std::endl;
                uint64_t flow_id = reversed_node_id + get_flow_id();
                write_flow_event(ss, parent->get_flow_us()+0.25, 's', "ControlFlow", flow_id,
                    saved_node_id, parent->thread_id, parent->task_id->get_name(), p->get_task_id()->get_name());
                write_flow_event(ss, p->get_start_us()-0.25, 'f', "ControlFlow", flow_id,
                    saved_node_id, _tid, parent->task_id->get_name(), p->get_task_id()->get_name());
            }
        }
        if (p->tt_ptr->explicit_trace_start) {
            ss << "{\"name\":\"" << p->get_task_id()->get_name()
              << "\",\"cat\":\"CPU\""
              << ",\"ph\":\"E\",\"pid\":"
              << saved_node_id << ",\"tid\":" << _tid
              << ",\"ts\":" << p->get_stop_us()
              << "},\n";
        } else {
            std::string pguid = parents_to_string(p->tt_ptr);
            ss << "{\"name\":\"" << p->get_task_id()->get_name()
              << "\",\"cat\":\"CPU\""
              << ",\"ph\":\"X\",\"pid\":"
              << saved_node_id << ",\"tid\":" << _tid
              << ",\"ts\":" << p->get_start_us() << ",\"dur\":"
              << p->get_stop_us() - p->get_start_us()
              << ",\"args\":{\"GUID\":" << p->guid << ",\"Parent GUID\":" << pguid << "}},\n";
        }
#if APEX_HAVE_PAPI
        int i = 0;
        for (auto metric :
            apex::instance()->the_profiler_listener->get_metric_names()) {
            double stop = p->papi_stop_values[i++];
            /* this would be a good idea, but Perfetto allows us to visualize
               as a delta or a rate, so not needed. It also confuses things for
               nested timers, so for now, just allow monotonically increasing
               counters to increase. */
            /*
            double start = p->papi_start_values[i];
            if (!p->tt_ptr->explicit_trace_start) {
                stop = stop - start;
            }
            */
            // write our counter into the event stream
            ss << fixed;
            ss << "{\"name\":\"" << metric
               << "\",\"cat\":\"CPU\""
               << ",\"ph\":\"C\",\"pid\":" << saved_node_id
               << ",\"ts\":" << p->get_stop_us()
               << ",\"args\":{\"value\":" << stop
               << "}},\n";
            //std::cout << p->get_task_id()->get_name() << "." << metric << ": " << start << " -> " << stop << " = " << stop-start << std::endl;
        }
#endif
        write_to_trace(ss);
    }
    flush_trace_if_necessary();
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
        ss.precision(3);
        ss << fixed;
        ss << "{\"name\":\"" << *(data.counter_name)
              << "\",\"cat\":\"CPU\""
              << ",\"ph\":\"C\",\"pid\":" << saved_node_id
              << ",\"ts\":" << profiler::now_us()
              << ",\"args\":{\"value\":" << data.counter_value
              << "}},\n";
        write_to_trace(ss);
    }
    flush_trace_if_necessary();
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

std::string trace_event_listener::make_tid (base_thread_node &node) {
    size_t tid;
    /* There is a potential for overlap here, but not a high potential.  The CPU and the GPU
     * would BOTH have to spawn 64k+ threads/streams for this to happen. */
    if (vthread_map.count(node) == 0) {
        uint32_t id_shifted = node.sortable_tid();
        vthread_map.insert(std::pair<base_thread_node, size_t>(node,id_shifted));
        std::stringstream ss;
        ss.precision(3);
        ss << fixed;
        ss << "{\"name\":\"thread_name\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_shifted
           << ",\"args\":{\"name\":\"";
        ss << node.name();
        //ss << "" << activity_to_string(node._activity);
        ss << "\"";
        ss << "}},\n";
        /* make sure the GPU threads come after CPU threads
         * by giving them a thread sort index of max int. */
        ss << "{\"name\":\"thread_sort_index\""
           << ",\"ph\":\"M\",\"pid\":" << saved_node_id
           << ",\"tid\":" << id_shifted
           << ",\"args\":{\"sort_index\":" << id_shifted << "}},\n";
        write_to_trace(ss);
    }
    tid = vthread_map[node];
    std::stringstream ss;
    ss << tid;
    std::string label{ss.str()};
    return label;
}

void trace_event_listener::on_async_event(base_thread_node &node,
    std::shared_ptr<profiler> &p, const async_event_data& data) {
    if (!_terminate) {
        std::stringstream ss;
        ss.precision(3);
        ss << fixed;
        std::string tid{make_tid(node)};
        std::string pguid = parents_to_string(p->tt_ptr);
        ss << "{\"name\":\"" << p->get_task_id()->get_name()
              << "\",\"cat\":\"GPU\""
              << ",\"ph\":\"X\",\"pid\":"
              << saved_node_id << ",\"tid\":" << tid
              << ",\"ts\":" << p->get_start_us() << ",\"dur\":"
              << p->get_stop_us() - p->get_start_us()
              << ",\"args\":{\"GUID\":" << p->guid << ",\"Parent GUID\":" << pguid << "}},\n";
        // write a flow event pair!
        // make sure the start of the flow is before the end of the flow, ideally the middle of the parent
        if (data.flow) {
            uint64_t flow_id = reversed_node_id + get_flow_id();
        if (data.reverse_flow) {
            double begin_ts = (p->get_stop_us() + p->get_start_us()) * 0.5;
            double end_ts = std::min(p->get_stop_us(), data.parent_ts_stop);
            write_flow_event(ss, begin_ts, 's', data.cat, flow_id, saved_node_id, atol(tid.c_str()), data.name, p->get_task_id()->get_name());
            write_flow_event(ss, end_ts, 't', data.cat, flow_id, saved_node_id, data.parent_tid, data.name, p->get_task_id()->get_name());
        } else {
            double begin_ts = std::min(p->get_start_us(), ((data.parent_ts_stop + data.parent_ts_start) * 0.5));
            double end_ts = p->get_start_us();
            write_flow_event(ss, begin_ts, 's', data.cat, flow_id, saved_node_id, data.parent_tid, data.name, p->get_task_id()->get_name());
            write_flow_event(ss, end_ts, 't', data.cat, flow_id, saved_node_id, atol(tid.c_str()), data.name, p->get_task_id()->get_name());
        }
        }
        write_to_trace(ss);
    }
    //flush_trace_if_necessary();
}

void trace_event_listener::on_async_metric(base_thread_node &node,
    std::shared_ptr<profiler> &p) {
    if (!_terminate) {
        std::stringstream ss;
        ss.precision(3);
        ss << fixed;
        std::string tid{make_tid(node)};
        ss << "{\"name\": \"" << p->get_task_id()->get_name()
              << "\",\"cat\":\"GPU\""
              << ",\"ph\":\"C\",\"pid\": " << saved_node_id
              << ",\"ts\":" << p->get_stop_us()
              << ",\"args\":{\"value\":" << p->value
              << "}},\n";
        write_to_trace(ss);
    }
    //flush_trace_if_necessary();
}

size_t trace_event_listener::get_thread_index(void) {
    static size_t numthreads{0};
    _vthread_mutex.lock();
    size_t tmpval = numthreads++;
    _vthread_mutex.unlock();
    return tmpval;
}

// Do we have just one stream, or a stream per thread?
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
    return trace;
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

std::string trace_event_listener::get_file_name() {
    saved_node_id = apex::instance()->get_node_id();
    std::stringstream ss;
    ss << apex_options::output_file_path() << "/";
    ss << "trace_events." << saved_node_id << ".json";
#ifdef APEX_HAVE_ZLIB
    ss << ".gz";
#endif
    std::string tmp{ss.str()};
    return tmp;
}

#ifdef APEX_HAVE_ZLIB
io::gzofstream& trace_event_listener::get_trace_file() {
    static io::gzofstream _trace_file(get_file_name());
    // automatically opens
    static bool header{false};
    if (!header) {
        header = true;
        _trace_file << fixed << "{\n";
        //_trace_file << "\"displayTimeUnit\": \"us\",\n";
        _trace_file << "\"traceEvents\": [\n";
    }
    return _trace_file;
}
#else
std::ofstream& trace_event_listener::get_trace_file() {
    static std::ofstream _trace_file;
    if (!_trace_file.is_open()) {
        _trace_file.open(get_file_name());
        _trace_file << fixed << "{\n";
        //_trace_file << "\"displayTimeUnit\": \"ms\",\n";
        _trace_file << "\"traceEvents\": [\n";
    }
    return _trace_file;
}
#endif

void trace_event_listener::flush_trace(trace_event_listener* listener) {
    static atomic<bool> flushing{false};
    // don't track memory in this new thread/function.
    in_apex prevent_memory_tracking;
    while (flushing) {} // wait for asynchronous flushing, if it's happening
    flushing = true;
    auto& trace_file = listener->get_trace_file();
#ifdef SERIAL
    // flush the trace
    listener->_vthread_mutex.lock();
    trace_file << listener->trace->rdbuf();
    // reset the buffer
    listener->trace->str("");
    listener->_vthread_mutex.unlock();
    /*
    // save the stringstream pointer
    std::stringstream* tmp = listener->trace;
    // make to new stringstream
    listener->trace = new std::stringstream{};
    // unlock and continue
    listener->_vthread_mutex.unlock();
    // flush the old stream to disk
    trace_file << tmp->rdbuf();
    // delete the flushed stringstream
    delete tmp;
    */
#else
    listener->_vthread_mutex.lock();
    size_t count = listener->streams.size();
    listener->_vthread_mutex.unlock();
    std::stringstream ss;
    for (size_t index = 0 ; index < count ; index++) {
        auto p2 = scoped_timer("APEX: " + to_string(index) + " thread flush");
        std::mutex * mtx = listener->get_thread_mutex(index);
        std::stringstream * strm = listener->get_thread_stream(index);
        mtx->lock();
        ss << strm->rdbuf();
        strm->str("");
        mtx->unlock();
    }
    // flush the trace
    trace_file << ss.rdbuf() << std::flush;
#endif
    // flush the trace
    trace_file << std::flush;
    flushing = false;
}

void trace_event_listener::flush_trace_if_necessary(bool force) {
    auto tmp = ++num_events;
    /* flush after every 100k events */
    if (tmp % 1000000 == 0 || force) {
        //flush_trace(this);
        //std::async(std::launch::async, flush_trace, this);
        //std::cout << "Flushing APEX trace..." << std::endl;
        std::thread(flush_trace, this).detach();
    }
}

void trace_event_listener::close_trace(void) {
    static bool closed{false};
    if (closed) return;
    auto& trace_file = get_trace_file();
    std::stringstream ss;
    ss.precision(3);
    ss << fixed;
    ss << "{\"name\":\"APEX Trace End\""
       << ", \"ph\":\"R\",\"pid\":"
       << saved_node_id << ",\"tid\":0,\"ts\":"
       << fixed << _end_time << "}\n";
    ss << "]\n";
    ss << "}\n" << std::endl;
    write_to_trace(ss);
    flush_trace(this);
    //printf("Closing trace...\n"); fflush(stdout);
    trace_file.close();
    closed = true;
}

/* This function is used by APEX threads so that TAU knows about them. */
int initialize_worker_thread_for_trace_event(void) {
    if (trace_event_listener::initialized())
    {
    }
    return 0;
}

}// end namespace

