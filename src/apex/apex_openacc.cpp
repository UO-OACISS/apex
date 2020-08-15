#include <iostream>
#include <sstream>
#include "apex.hpp"
#include "apex_openacc.hpp"

/* Function called by OpenACC runtime to register callbacks */
extern "C" {

void acc_register_library(acc_prof_reg reg,
    acc_prof_reg unreg, APEX_OPENACC_LOOKUP_FUNC lookup) {
    DEBUG_PRINT("Inside acc_register_library\n");
    APEX_UNUSED(unreg);
    APEX_UNUSED(lookup);

    // Launch events
    reg( acc_ev_enqueue_launch_start,      apex_openacc_launch_callback, acc_reg );
    reg( acc_ev_enqueue_launch_end,        apex_openacc_launch_callback, acc_reg );

    /* The data events aren't signal safe, for some reason.  So, that means we can't
     * allocate any memory, which limits what we can do.  For that reason, we only
     * handle some events, and with static timer names. */
    // Data events
    reg( acc_ev_enqueue_upload_start,      apex_openacc_data_callback, acc_reg );
    reg( acc_ev_enqueue_upload_end,        apex_openacc_data_callback, acc_reg );
    reg( acc_ev_enqueue_download_start,    apex_openacc_data_callback, acc_reg );
    reg( acc_ev_enqueue_download_end,      apex_openacc_data_callback, acc_reg );
    reg( acc_ev_create,                    apex_openacc_data_callback, acc_reg );
    reg( acc_ev_delete,                    apex_openacc_data_callback, acc_reg );
    reg( acc_ev_alloc,                     apex_openacc_data_callback, acc_reg );
    reg( acc_ev_free,                      apex_openacc_data_callback, acc_reg );
    // Other events
    reg( acc_ev_device_init_start,         apex_openacc_other_callback, acc_reg );
    reg( acc_ev_device_init_end,           apex_openacc_other_callback, acc_reg );
    reg( acc_ev_device_shutdown_start,     apex_openacc_other_callback, acc_reg );
    reg( acc_ev_device_shutdown_end,       apex_openacc_other_callback, acc_reg );
    reg( acc_ev_runtime_shutdown,          apex_openacc_other_callback, acc_reg );
    reg( acc_ev_enter_data_start,          apex_openacc_other_callback, acc_reg );
    reg( acc_ev_enter_data_end,            apex_openacc_other_callback, acc_reg );
    reg( acc_ev_exit_data_start,           apex_openacc_other_callback, acc_reg );
    reg( acc_ev_exit_data_end,             apex_openacc_other_callback, acc_reg );
    reg( acc_ev_update_start,              apex_openacc_other_callback, acc_reg );
    reg( acc_ev_update_end,                apex_openacc_other_callback, acc_reg );
    reg( acc_ev_compute_construct_start,   apex_openacc_other_callback, acc_reg );
    reg( acc_ev_compute_construct_end,     apex_openacc_other_callback, acc_reg );
    reg( acc_ev_wait_start,                apex_openacc_other_callback, acc_reg );
    reg( acc_ev_wait_end,                  apex_openacc_other_callback, acc_reg );

} // acc_register_library

void apex_read_api_info(acc_api_info* api_info, bool context) {
    static bool first{true};
    if (first) {
        std::cout << "Device API: " << acc_device_api_names[api_info->device_api];
        std::cout << " Device type: ";
        switch (api_info->device_type) {
            case acc_device_current:
                std::cout << "current";
                break;
            case acc_device_none:
                std::cout << "none";
                break;
            case acc_device_default:
                std::cout << "default";
                break;
            case acc_device_host:
                std::cout << "host";
                break;
            case acc_device_not_host:
                std::cout << "not_host";
                break;
            case acc_device_nvidia:
                std::cout << "nvidia";
                break;
            case acc_device_radeon:
                std::cout << "radeon";
                break;
            default:
                std::cout << "unknown";
                break;
        }
        std::cout << " Device vendor: " << api_info->vendor << std::endl;
    }
    if (context && first) {
        std::cout << "Device Handle: " << api_info->device_handle;
        std::cout << " Context: " << api_info->context_handle;
        std::cout << " Queue/Stream: " << api_info->async_handle << std::endl;
        //first = false;
    }
}

void apex_openacc_launch_callback(acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info) {
    apex_read_api_info(api_info, true);
    acc_launch_event_info* launch_event = &(event_info->launch_event);
    std::stringstream ss;
    ss << "OpenACC enqueue launch: ";

    /* if this is an end event, short circuit by grabbing the FunctionInfo pointer out
     * of tool_info and stopping that timer; if the pointer is NULL something bad
     * happened, print warning and kill whatever timer is on top of the stack
     */
    if (prof_info->event_type == acc_ev_enqueue_launch_end) {
        if (launch_event->tool_info == NULL) {
            fprintf(stderr, "WARNING: OpenACC launch end event has bad matching start event.");
            apex::stop(apex::thread_instance::instance().get_current_profiler());
        }
        else {
            apex::profiler * p = (apex::profiler*)(launch_event->tool_info);
            apex::stop(p);
        }
        return;
    }

    ss << launch_event->kernel_name;

    if (launch_event->implicit) {
        ss << " (implicit)";
    }
    if (launch_event->parent_construct < 9999) {
        ss << " "
           << acc_constructs[launch_event->parent_construct];
    }

    if (prof_info->src_file != nullptr &&
        prof_info->line_no != -1 &&
        prof_info->end_line_no != -1) {
        ss << " [{"
            << prof_info->src_file
            << "} {"
            << prof_info->line_no
            << ","
            << prof_info->end_line_no
            << "}]";
    }

    /* if this is a start event, get the FunctionInfo and put it in tool_info
     * so the end event will get it to stop the timer
     */
    std::string tmp{ss.str()};
    if (prof_info->event_type == acc_ev_enqueue_launch_start) {
        void* func_info = (void*)apex::start(tmp);
        launch_event->tool_info = func_info;
        apex::sample_value("OpenACC Gangs", launch_event->num_gangs);
        apex::sample_value("OpenACC Workers", launch_event->num_workers);
        apex::sample_value("OpenACC Vector Lanes", launch_event->vector_length);
    }
    else {
        apex::sample_value(tmp, 1);
    }
}

void apex_openacc_other_callback( acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info ) {
    APEX_UNUSED(api_info);
    acc_other_event_info* other_event = &(event_info->other_event);
    int start = -1;
    std::stringstream ss;

    switch(prof_info->event_type) {
        case acc_ev_device_init_start:
            start = 1;
            ss << "OpenACC device init";
            apex_read_api_info(api_info, false);
            break;
        case acc_ev_device_init_end:
            start = 0;
            break;
        case acc_ev_device_shutdown_start:
            start = 1;
            ss << "OpenACC device shutdown";
            break;
        case acc_ev_device_shutdown_end:
            start = 0;
            break;
        case acc_ev_runtime_shutdown:
            start = -1;
            ss << "OpenACC runtime shutdown";
            break;
        case acc_ev_enter_data_start:
            start = 1;
            ss << "OpenACC enter data";
            break;
        case acc_ev_enter_data_end:
            start = 0;
            break;
        case acc_ev_exit_data_start:
            start = 1;
            ss << "OpenACC exit data";
            break;
        case acc_ev_exit_data_end:
            start = 0;
            break;
        case acc_ev_update_start:
            start = 1;
            ss << "OpenACC update";
            break;
        case acc_ev_update_end:
            start = 0;
            break;
        case acc_ev_compute_construct_start:
            start = 1;
            ss << "OpenACC compute construct";
            break;
        case acc_ev_compute_construct_end:
            start = 0;
            break;
        case acc_ev_wait_start:
            start = 1;
            ss << "OpenACC wait";
            break;
        case acc_ev_wait_end:
            start = 0;
            break;
        default:
            start = -1;
            ss << "UNKNOWN OPENACC OTHER EVENT";
            fprintf(stderr, "ERROR: Unknown other event passed to OpenACC other event callback.");
    }

    /* if this is an end event, short circuit by grabbing the FunctionInfo
     * pointer out of tool_info and stopping that timer; if the pointer is
     * NULL something bad happened, print warning and kill whatever timer
     * is on top of the stack
     */
    if (start == 0) {
        if (other_event->tool_info == NULL) {
            apex::stop(apex::thread_instance::instance().get_current_profiler());
        }
        else {
            apex::profiler * p = (apex::profiler*)(other_event->tool_info);
            apex::stop(p);
        }
        return;
    }

    if (other_event->implicit) {
        ss << " (implicit)";
    }
    if (other_event->parent_construct < 9999) {
        ss << " "
            << acc_constructs[other_event->parent_construct];
    }

    if (prof_info->src_file != nullptr &&
        prof_info->line_no != -1 &&
        prof_info->end_line_no != -1) {
        ss << " [{"
            << prof_info->src_file
            << "} {"
            << prof_info->line_no
            << ","
            << prof_info->end_line_no
            << "}]";
    }

    std::string tmp{ss.str()};
    if (start == 1) {
        void* func_info = (void*)apex::start(tmp);
        other_event->tool_info = func_info;
    }
    else {
        apex::sample_value(tmp, 1);
    }
}

void apex_openacc_data_callback(acc_prof_info* prof_info,
    acc_event_info* event_info, acc_api_info* api_info ) {
    acc_data_event_info* data_event = &(event_info->data_event);
    APEX_UNUSED(api_info);
    int start = -1;
    std::stringstream ss;

    switch(prof_info->event_type) {
        case acc_ev_enqueue_upload_start:
            start = 1;
            ss << "OpenACC enqueue data transfer (HtoD)";
            break;
        case acc_ev_enqueue_upload_end:
            start = 0;
            break;
        case acc_ev_enqueue_download_start:
            start = 1;
            ss << "OpenACC enqueue data transfer (DtoH)";
            break;
        case acc_ev_enqueue_download_end:
            start = 0;
            break;
        case acc_ev_create:
            start = -1;
            ss << "OpenACC device data create";
            break;
        case acc_ev_delete:
            start = -1;
            ss << "OpenACC device data delete";
            break;
        case acc_ev_alloc:
            start = -1;
            ss << "OpenACC device alloc";
            break;
        case acc_ev_free:
            start = -1;
            ss << "OpenACC device free";
            break;
        default:
            start = -1;
            ss << "UNKNOWN OPENACC DATA EVENT";
    }

    /* if this is an end event, short circuit by grabbing the FunctionInfo
     * pointer out of tool_info and stopping that timer; if the pointer is
     * NULL something bad happened, print warning and kill whatever timer
     * is on top of the stack
     */
    if (start == 0) {
        if (data_event->tool_info == NULL) {
            apex::stop(apex::thread_instance::instance().get_current_profiler());
        }
        else {
            apex::profiler * p = (apex::profiler*)(data_event->tool_info);
            apex::stop(p);
        }
        return;
    }

    if (data_event->var_name != nullptr) {
        ss << " " << data_event->var_name;
    }
    if (data_event->implicit) {
        ss << " (implicit)";
    }
    if (data_event->parent_construct < 9999) {
        ss << " "
            << acc_constructs[data_event->parent_construct];
    }

    if (prof_info->src_file != nullptr &&
        prof_info->line_no != -1 &&
        prof_info->end_line_no != -1) {
        ss << " [{"
            << prof_info->src_file
            << "} {"
            << prof_info->line_no
            << ","
            << prof_info->end_line_no
            << "}]";
    }

    if (start == 1) {
        {
            std::string tmp{ss.str()};
            void* func_info = (void*)apex::start(tmp);
            data_event->tool_info = func_info;
        }
        ss << " Bytes";
        std::string tmp{ss.str()};
        double bytes = (double)data_event->bytes;
        apex::sample_value(tmp, bytes);
    }
    else {
        ss << " Bytes";
        std::string tmp{ss.str()};
        double bytes = 0.0;
        if (data_event->event_type == acc_ev_create ||
            data_event->event_type == acc_ev_alloc) {
            bytes = (double)data_event->bytes;
        }
        apex::sample_value(tmp, bytes);
    }
}


} // extern "C"
