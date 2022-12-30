/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "perfetto_static.hpp"
#include "perfetto_listener.hpp"
#include "thread_instance.hpp"
#include "apex.hpp"
#include <fstream>

using namespace std;
constexpr const char * _category{"APEX"};
constexpr const char * _guid{"GUID"};
constexpr const char * _pguid{"parent GUID"};

namespace apex {

void perfetto_listener::get_file_name() {
    auto saved_node_id = apex::instance()->get_node_id();
    std::stringstream ss;
    ss << apex_options::output_file_path() << "/";
    ss << "trace_events." << saved_node_id << ".pftrace";
    filename = ss.str();
}

perfetto_listener::perfetto_listener (void) {
    perfetto::TracingInitArgs args;
    // The backends determine where trace events are recorded. For this example we
    // are going to use the in-process tracing service, which only includes in-app
    // events.
    args.backends = perfetto::kInProcessBackend;

    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    // The trace config defines which types of data sources are enabled for
    // recording. In this example we just need the "track_event" data source,
    // which corresponds to the TRACE_EVENT trace points.
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(1024);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    tracing_session = perfetto::Tracing::NewTrace();
    get_file_name();
    int file_descriptor = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    tracing_session->Setup(cfg, file_descriptor);
    tracing_session->StartBlocking();
}

perfetto_listener::~perfetto_listener (void) {
    // Make sure the last event is closed for this example.
    perfetto::TrackEvent::Flush();

    // Stop tracing and read the trace data.
    tracing_session->StopBlocking();
    /*
    std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

    // Write the result into a file.
    // Note: To save memory with longer traces, you can tell Perfetto to write
    // directly into a file by passing a file descriptor into Setup() above.
    std::ofstream output;
    output.open("example.pftrace", std::ios::out | std::ios::binary);
    output.write(&trace_data[0], std::streamsize(trace_data.size()));
    output.close();
    PERFETTO_LOG(
        "Trace written in example.pftrace file. To read this trace in "
        "text form, run `./tools/traceconv text example.pftrace`");
        */
    close(file_descriptor);
}

void perfetto_listener::on_startup(startup_event_data &data) {
    APEX_UNUSED(data);
    // Give a custom name for the traced process.
    perfetto::ProcessTrack process_track = perfetto::ProcessTrack::Current();
    perfetto::protos::gen::TrackDescriptor desc = process_track.Serialize();
    desc.mutable_process()->set_process_name(thread_instance::program_path());
    perfetto::TrackEvent::SetTrackDescriptor(process_track, desc);
    return;
}

void perfetto_listener::on_dump(dump_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::on_pre_shutdown(void) {
    return;
}

void perfetto_listener::on_shutdown(shutdown_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::on_new_node(node_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::on_new_thread(new_thread_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::on_exit_thread(event_data &data) {
    APEX_UNUSED(data);
    return;
}

inline bool perfetto_listener::_common_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    APEX_UNUSED(tt_ptr);
    TRACE_EVENT_BEGIN(_category,
        perfetto::DynamicString{tt_ptr->get_task_id()->get_name()},
        //perfetto::ProcessTrack::Current(),
        (uint64_t)tt_ptr->prof->get_start_ns(),
        _guid, tt_ptr->guid,
        _pguid, tt_ptr->parent_guid);
    return true;
}

inline void perfetto_listener::_common_stop(std::shared_ptr<profiler> &p) {
    APEX_UNUSED(p);
    TRACE_EVENT_END(_category,
        //perfetto::ProcessTrack::Current(),
        (uint64_t)p->get_stop_ns());
    return;
}

bool perfetto_listener::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    return _common_start(tt_ptr);
}

bool perfetto_listener::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
    return _common_start(tt_ptr);
}

void perfetto_listener::on_stop(std::shared_ptr<profiler> &p) {
    return _common_stop(p);
}

void perfetto_listener::on_yield(std::shared_ptr<profiler> &p) {
    return _common_stop(p);
}

void perfetto_listener::on_sample_value(sample_value_event_data &data) {
    APEX_UNUSED(data);
    TRACE_COUNTER(_category,
        data.counter_name->c_str(),
        (uint64_t)profiler::now_ns(),
        data.counter_value);
    return;
}

void perfetto_listener::on_periodic(periodic_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::on_custom_event(custom_event_data &data) {
    APEX_UNUSED(data);
    return;
}

void perfetto_listener::set_node_id(int node_id, int node_count) {
    APEX_UNUSED(node_id);
    APEX_UNUSED(node_count);
}

size_t perfetto_listener::make_tid (base_thread_node &node) {
    size_t tid;
    /* There is a potential for overlap here, but not a high potential.  The CPU and the GPU
     * would BOTH have to spawn 64k+ threads/streams for this to happen. */
    _vthread_mutex.lock();
    if (vthread_map.count(node) == 0) {
        uint32_t id_shifted = node.sortable_tid();
        vthread_map.insert(std::pair<base_thread_node, size_t>(node,id_shifted));
    }
    tid = vthread_map[node];
    _vthread_mutex.unlock();
    return tid;
}

void perfetto_listener::on_async_event(base_thread_node &node,
    std::shared_ptr<profiler> &p, const async_event_data& data) {
    const size_t tid{make_tid(node)};
    uint64_t pguid = 0;
    if (p->tt_ptr != nullptr && p->tt_ptr->parent != nullptr) {
        pguid = p->tt_ptr->parent->guid;
    }
    TRACE_EVENT_BEGIN(_category,
        perfetto::DynamicString{p->get_task_id()->get_name()},
        perfetto::Track(tid),
        (uint64_t)p->get_start_ns(), _guid, p->guid, _pguid, pguid);
    TRACE_EVENT_END(_category,
        perfetto::Track(tid),
        (uint64_t)p->get_stop_ns());
    APEX_UNUSED(data);
}

void perfetto_listener::on_async_metric(base_thread_node &node,
    std::shared_ptr<profiler> &p) {
    APEX_UNUSED(node);
    TRACE_COUNTER(_category,
        p->get_task_id()->get_name().c_str(),
        (uint64_t)p->get_stop_ns(),
        p->value);
}

}// end namespace

