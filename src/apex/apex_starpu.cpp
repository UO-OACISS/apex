#include <starpu.h>
#include <assert.h>
#include <inttypes.h>
#include "apex.hpp"
#include <starpu_profiling_tool.h>

void starpu_perf_counter_collection_start( void );
void starpu_perf_counter_collection_stop( void );

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#warning "COMPILING STARPU SUPPORT"
/* global counters */
static int id_g_total_submitted;
static int id_g_peak_submitted;
static int id_g_peak_ready;

/* per worker counters */
static int id_w_total_executed;
static int id_w_cumul_execution_time;

std::map<int,std::string> dev_type;

extern "C" {

    /* Functions plugged into the scheduler's callbacks 
       These functions come from perf_counters_02.c, I simply removed the one
       related to codelets, because I have no codelet to tie it to.
    */
    
    void g_listener_cb(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context) {
        (void) listener;
        (void) context;
        int64_t g_total_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_g_total_submitted);
        int64_t g_peak_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_g_peak_submitted);
        int64_t g_peak_ready = starpu_perf_counter_sample_get_int64_value(sample, id_g_peak_ready);
        //   printf("global: g_total_submitted = %"PRId64", g_peak_submitted = %"PRId64", g_peak_ready = %"PRId64"\n", g_total_submitted, g_peak_submitted, g_peak_ready);
        
        std::stringstream ss1;
        ss1 << "Total submitted tasks (g_total_submitted)";
        std::string tmp1{ss1.str()};
        apex::sample_value( tmp1, g_total_submitted );

        std::stringstream ss2;
        ss2 << "Peak number of tasks submitted (g_peak_submitted)";
        std::string tmp2{ss2.str()};
        apex::sample_value( tmp2, g_peak_submitted );

        std::stringstream ss3;
        ss3 << "Peak number of ready tasks (g_peak_ready)";
        std::string tmp3{ss3.str()};
        apex::sample_value( tmp3, g_peak_ready );
    }
    
    void w_listener_cb(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context) {
        (void) listener;
        (void) context;
        int workerid = starpu_worker_get_id();
        int64_t w_total_executed = starpu_perf_counter_sample_get_int64_value(sample, id_w_total_executed);
        double w_cumul_execution_time = starpu_perf_counter_sample_get_double_value(sample, id_w_cumul_execution_time);
        
        //  printf("worker[%d]: w_total_executed = %"PRId64", w_cumul_execution_time = %lf\n", workerid, w_total_executed, w_cumul_execution_time);

        std::stringstream ss1;
        ss1 << "Worker " << workerid << " w_total_executed";
        std::string tmp1{ss1.str()};
        apex::sample_value( tmp1, w_total_executed );

        std::stringstream ss2;
        ss2 <<  "Worker " << workerid << " w_cumul_execution_time";
        std::string tmp2{ss2.str()};
        apex::sample_value( tmp2, w_cumul_execution_time );
    }
    
    const enum starpu_perf_counter_scope g_scope = starpu_perf_counter_scope_global;
    const enum starpu_perf_counter_scope w_scope = starpu_perf_counter_scope_per_worker;
    
    struct starpu_perf_counter_set *g_set;
    struct starpu_perf_counter_set *w_set;
    struct starpu_perf_counter_listener * g_listener;
    struct starpu_perf_counter_listener * w_listener;
    
    /******************************************************************************/
    /*                           My callbacks.                                    */
    /******************************************************************************/
    
    /* This one must be called at the very beginning of the initialization, otherwise
       the flag might be enabled too late to be taken into account.
    */

    void enable_counters(starpu_prof_tool_info* prof_info, starpu_prof_tool_event_info* event_info, starpu_prof_tool_api_info* api_info ) {
    }
    
    /* This one is called at the end of the initialization.
       It gets a starpu_conf from the prof_info passed to the callback.
       It initializes the scheduler's internal callbacks.
    */
    
    void init_counters(starpu_prof_tool_info* prof_info, starpu_prof_tool_event_info* event_info, starpu_prof_tool_api_info* api_info ) {
        
        struct starpu_conf *conf = prof_info->conf;
        // printf( "%p\n", conf );
        
        /* Start collecting perfomance counter right after initialization */
        //	conf->start_perf_counter_collection = 1;
        starpu_perf_counter_collection_start();
        
        
        g_set = starpu_perf_counter_set_alloc(g_scope);
        STARPU_ASSERT(g_set != NULL);
        w_set = starpu_perf_counter_set_alloc(w_scope);
        STARPU_ASSERT(w_set != NULL);
        
        id_g_total_submitted = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_total_submitted");
        STARPU_ASSERT(id_g_total_submitted != -1);
        id_g_peak_submitted = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_peak_submitted");
        STARPU_ASSERT(id_g_peak_submitted != -1);
        id_g_peak_ready = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_peak_ready");
        STARPU_ASSERT(id_g_peak_ready != -1);
        
        id_w_total_executed = starpu_perf_counter_name_to_id(w_scope, "starpu.task.w_total_executed");
        STARPU_ASSERT(id_w_total_executed != -1);
        id_w_cumul_execution_time = starpu_perf_counter_name_to_id(w_scope, "starpu.task.w_cumul_execution_time");
        STARPU_ASSERT(id_w_cumul_execution_time != -1);
        
        starpu_perf_counter_set_enable_id(g_set, id_g_total_submitted);
        starpu_perf_counter_set_enable_id(g_set, id_g_peak_submitted);
        starpu_perf_counter_set_enable_id(g_set, id_g_peak_ready);
        
        starpu_perf_counter_set_enable_id(w_set, id_w_total_executed);
        starpu_perf_counter_set_enable_id(w_set, id_w_cumul_execution_time);
        
        g_listener = starpu_perf_counter_listener_init(g_set, g_listener_cb, (void *)(uintptr_t)42);
        w_listener = starpu_perf_counter_listener_init(w_set, w_listener_cb, (void *)(uintptr_t)17);
        
        starpu_perf_counter_set_global_listener(g_listener);
        starpu_perf_counter_set_all_per_worker_listeners(w_listener);
    }
    
    /* This one is called when StarPU is being finalized. 
       It is required to clean-up the callbacks.
    */
    
    void finalize_counters( starpu_prof_tool_info* prof_info, starpu_prof_tool_event_info* event_info, starpu_prof_tool_api_info* api_info ) {
        
        starpu_perf_counter_unset_all_per_worker_listeners();
        starpu_perf_counter_unset_global_listener();
        
        starpu_perf_counter_listener_exit(w_listener);
        starpu_perf_counter_listener_exit(g_listener);
        
        starpu_perf_counter_set_disable_id(w_set, id_w_cumul_execution_time);
        starpu_perf_counter_set_disable_id(w_set, id_w_total_executed);
        
        starpu_perf_counter_set_disable_id(g_set, id_g_peak_ready);
        starpu_perf_counter_set_disable_id(g_set, id_g_peak_submitted);
        starpu_perf_counter_set_disable_id(g_set, id_g_total_submitted);
        
        starpu_perf_counter_set_free(w_set);
        w_set = NULL;
        
        starpu_perf_counter_set_free(g_set);
        g_set = NULL;
        
        starpu_perf_counter_collection_stop();
        
    }
    
    /******************************************************************************/
    
    void myfunction_cb( struct starpu_prof_tool_info* prof_info,  union starpu_prof_tool_event_info* event_info, struct starpu_prof_tool_api_info* api_info ){
        
    std::stringstream event_name;
    std::stringstream info;
    const char* name;
    bool start = 0;
    
    int tag = 0;
    std::pair<std::string, std::string> fun;
    /*
    if( nullptr != prof_info->fun_ptr ){
        fun = funcname( prof_info->fun_ptr );
        } else*/ {
        fun.first = "";
        fun.second = "";
        }

    switch(  prof_info->event_type ) {
    case starpu_prof_tool_event_init:
        event_name << "StarPU";
        start = 1;
        break;
    case starpu_prof_tool_event_terminate:
        event_name << "StarPU";
        start = 0;
        break;
    case starpu_prof_tool_event_init_begin:
        event_name << "StarPU init";
        start = 1;
        break;
    case starpu_prof_tool_event_init_end:
        event_name << "StarPU init";
        start = 0;
        break;
    case starpu_prof_tool_event_driver_init:
        event_name << "StarPU driver";
        info << " " << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "";
        start = 1;
       break;       
    case starpu_prof_tool_event_driver_deinit:
        event_name << "StarPU driver";
        info << " " << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "";
        start = 0;
        break;
    case starpu_prof_tool_event_driver_init_start:
        event_name << "StarPU driver init";
        info << " " << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "";
        start = 1;
        break;
    case starpu_prof_tool_event_driver_init_end:
        event_name << "StarPU driver init";
        info << " " << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "";
        start = 0;
        break;
    case starpu_prof_tool_event_start_gpu_exec:
        event_name << "GPU: ";
    case starpu_prof_tool_event_start_cpu_exec:
        event_name << "StarPU exec ";
       info << " " << dev_type[prof_info->driver_type] << ":" << prof_info->device_number << " UNRESOLVED ADDR " << prof_info->fun_ptr << "  " << fun.second.c_str() << " ";
        start = 1;
        break;
    case starpu_prof_tool_event_end_gpu_exec:
        event_name << "GPU: ";
    case starpu_prof_tool_event_end_cpu_exec:
        event_name << "StarPU exec ";
        start = 0;
        info << fun.first.c_str() << "" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number << " UNRESOLVED ADDR " << prof_info->fun_ptr << "  " << fun.second.c_str() << " ";
        break;
    case starpu_prof_tool_event_start_transfer:
        event_name << "StarPU_transfer";
        info << "  memnode " << prof_info->memnode << " ";
        start = 1;
        break;
    case starpu_prof_tool_event_end_transfer:
        event_name << "StarPU_transfer";
        // TAU_TRACE_RECVMSG( tag, prof_info->memnode, prof_info->bytes_transfered );
        info << "  memnode " << prof_info->memnode << " ";
        start = 0;
        break;
    default:
        std::cout <<  "Unknown callback " <<  prof_info->event_type << std::endl;
        break;
    }
    
    event_name << info.str();
    // event_name << "UNRESOLVED ADDR 0x" << std::hex << prof_info->fun_ptr;
    // std::cout << "Event: " << event_name.str().c_str() << " " << event_name.str()[0] << std::endl;
    // std::cout << "Event1: " << event_name.str().c_str() << std::endl;

    if ( start == 1 ) {
        apex::start( event_name.str() );
    }  else if ( start == 0 ) {
        apex::stop(apex::thread_instance::instance().get_current_profiler());
    } else {
        /// TAU_VERBOSE("event_name = %s\n", &(event_name.str()[0]) );
    }
}
    
    /******************************************************************************/
    /*           Initialize myself as an external library.                       */
    /******************************************************************************/
    
    /* Register the callbacks */
    
 void starpu_prof_tool_library_register(starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg)
 {
    enum  starpu_prof_tool_command info;
        /* This one must be called at the *beginning* of the initialization
           Otherwise the flag might be set too late */
        reg( starpu_prof_tool_event_init_begin, &enable_counters, info );
        /* This one must be called at the *end* of the initialization
           Otherwise the counters might not be ready yet */
        reg( starpu_prof_tool_event_init_end, &init_counters, info );
        /* This one must be called at the end, but I don't know precisely when yet */
        reg( starpu_prof_tool_event_terminate, &finalize_counters, info );

        reg( starpu_prof_tool_event_init_begin, &myfunction_cb, info );
        //        reg( starpu_prof_tool_event_init_end, &myfunction_cb, info );
        reg( starpu_prof_tool_event_init, &myfunction_cb, info );
        //        reg( starpu_prof_tool_event_terminate, &myfunction_cb, info );
        reg( starpu_prof_tool_event_driver_init, &myfunction_cb, info );
        reg( starpu_prof_tool_event_driver_deinit, &myfunction_cb, info );
        reg( starpu_prof_tool_event_driver_init_start, &myfunction_cb, info );
        reg( starpu_prof_tool_event_driver_init_end, &myfunction_cb, info );
        reg( starpu_prof_tool_event_start_cpu_exec, &myfunction_cb, info );
        reg( starpu_prof_tool_event_end_cpu_exec, &myfunction_cb, info );
        reg( starpu_prof_tool_event_start_gpu_exec, &myfunction_cb, info );
        reg( starpu_prof_tool_event_end_gpu_exec, &myfunction_cb, info );
        reg( starpu_prof_tool_event_start_transfer, &myfunction_cb, info );
        reg( starpu_prof_tool_event_end_transfer, &myfunction_cb, info );
        
    }
    
} // extern "C"
