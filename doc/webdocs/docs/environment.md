# APEX Runtime Options

## Environment Variables

There are a number of environment variables that control APEX behavior
at runtime. The variables can be defined in the environment before
application execution, or specified in a file called `apex.conf` in the
current execution directory.  The format of the configuration file is:

```
APEX_VARIABLE1=value
APEX_VARIABLE2=value
...
```

To generate a default APEX configuration file in the current working directory, run the `./install/bin/apex_make_default_config` program.

| Environment Variable | Default Value | Valid Values | Description |
| -------------------- | -- | -- | -------------------------------- |
| `APEX_DISABLE` | 0 | 0,1 | Disable APEX during the application execution |
| `APEX_SUSPEND` | 0 | 0,1 | Suspend APEX timers and counters during the application execution |
| `APEX_PAPI_SUSPEND` | 0 | 0,1 | Suspend PAPI counters during the application execution |
| `APEX_SCREEN_OUTPUT` | 0 | 0,1 | Output APEX performance summary at exit |
| `APEX_VERBOSE` | 0 | 0,1 | Output APEX options at entry |
| `APEX_PROFILE_OUTPUT` | 0 | 0,1 | Output TAU profile of performance summary |
| `APEX_CSV_OUTPUT` | 0 | 0,1 | Output CSV profile of performance summary |
| `APEX_TASKGRAPH_OUTPUT` | 0 | 0,1 | Output graphviz reduced taskgraph |
| `APEX_POLICY` | 1 | 0,1 | Enable APEX policy listener and execute registered policies |
| `APEX_PROC_STAT` | 1 | 0,1 | Periodically read data from /proc/stat |
| `APEX_PROC_CPUINFO` | 0 | 0,1 | Read data (once) from /proc/cpuinfo |
| `APEX_PROC_MEMINFO` | 0 | 0,1 | Periodically read data from /proc/meminfo |
| `APEX_PROC_NET_DEV` | 0 | 0,1 | Periodically read data from /proc/net/dev |
| `APEX_PROC_SELF_STATUS` | 0 | 0,1 | Periodically read data from /proc/self/status |
| `APEX_PROC_SELF_IO` | 0 | 0,1 | Periodically read data from /proc/self/io |
| `APEX_PROC_STAT_DETAILS` | 0 | 0,1 | Periodically read detailed data from /proc/self/stat |
| `APEX_PROC_PERIOD` | 1000000 | Integer | /proc data read sampling period, in microseconds |
| `APEX_MEASURE_CONCURRENCY` | 0 | 0,1 | Periodically sample thread activity and output report at exit |
| `APEX_MEASURE_CONCURRENCY_PERIOD` | 1000000 | Integer | Thread concurrency sampling period, in microseconds |
| `APEX_OTF2` | 0 | 0,1 | Enable OTF2 trace output. |
| `APEX_TRACE_EVENT` | 0 | 0,1 | Enable Google Trace Event output. |
| `APEX_OTF2_ARCHIVE_PATH` | `OTF2_archive` | valid path | OTF2 trace directory. |
| `APEX_OTF2_ARCHIVE_NAME` | `APEX` | valid string | OTF2 trace filename. |
| `APEX_TAU` | 0 | 0,1 | Enable TAU profiling (if application is executed with `tau_exec`). |
| `APEX_THROTTLE_CONCURRENCY` | 0 | 0,1 | Enable thread concurrency throttling |
| `APEX_THROTTLING_MIN_THREADS` | 1 | 0,1 | Minimum threads allowed |
| `APEX_THROTTLING_MAX_THREADS` | 8 | 0,1 | Maximum threads allowed |
| `APEX_THROTTLE_ENERGY` | 0 | 0,1 | Enable energy throttling |
| `APEX_THROTTLE_ENERGY_PERIOD` | 1000000 | Integer | Power sampling period, in microseconds |
| `APEX_THROTTLING_MIN_WATTS` | 150 | Integer | Minimum Watt threshold |
| `APEX_THROTTLING_MAX_WATTS` | 300 | Integer | Maximum Watt threshold |
| `APEX_PTHREAD_WRAPPER_STACK_SIZE` | 0 | 16k-8M | When wrapping pthread_create, use this size for the stack. |
| `APEX_PAPI_METRICS` | *null* | space-delimited string of metric names | List of metrics to be measured by APEX when timers are used. Only meaningful if APEX is configured with PAPI support.  Any supported metric from *papi_avail* ([see PAPI Documentation](http://icl.cs.utk.edu/projects/papi/wiki/PAPIC:papi_avail.1)) can be used. |
| `APEX_PAPI_SUSPEND` | 0 | 0,1 | Suspend collection of PAPI metrics for APEX timers during the application execution |
| `APEX_PROCESS_ASYNC_STATE` | 1 | 0,1 | Enable/disable asynchronous processing of statistics (useful when only collecting trace data) |
| `APEX_UNTIED_TIMERS` | 0 | 0,1 | Disable callstack state maintenance for specific OS threads.  This allows APEX timers to start on one thread and stop on another.  This is not compatible with tracing. |
| `APEX_OMPT_REQUIRED_EVENTS_ONLY` | 0 | 0,1 | Disable moderate-frequency, moderate-overhead OMPT events. |
| `APEX_OMPT_HIGH_OVERHEAD_EVENTS` | 0 | 0,1 | Disable high-frequency, high-overhead OMPT events. |
| `APEX_PIN_APEX_THREADS` | 1 | 0,1 | Pin APEX asynchronous threads to the last core/PU on the system. |
| `APEX_TASK_SCATTERPLOT` | 0 | 0,1 | Periodically sample APEX tasks, generating a scatterplot of time distributions. |
| `APEX_TIME_TOP_LEVEL_OS_THREADS` | 0 | 0,1 | When registering threads, measure their lifetimes. |
| `APEX_CUDA_COUNTERS` | 0 | 0,1 | Enable CUDA CUPTI counter measurement. |
| `APEX_CUDA_KERNEL_DETAILS` | 0 | 0,1 | Enable Context information for CUDA CUPTI counter measurement and CUDA CUPTI API callback timers. |
| `APEX_CUDA_RUNTIME_API` | 1 | 0,1 | Enable callbacks for the CUDA Runtime API (`cuda*()` functions). |
| `APEX_CUDA_DRIVER_API` | 0 | 0,1 | Enable callbacks for the CUDA Driver API (`cu*()` functions). |
| `APEX_JUPYTER_SUPPORT` | 0 | 0,1 | When running HPX in a Jupyter notebook, enable special handling for APEX data output and system reset. |

## `apex_exec` flags

To control the behavior of APEX when using `apex_exec`, many flags are available, several of which will automatically set the above environment variables as necessary:

```
Usage:
apex_exec <APEX options> executable <executable options>

where APEX options are zero or more of:
    --apex:help            show this usage message
    --apex:debug           run with APEX in debugger
    --apex:verbose         enable verbose list of APEX environment variables
    --apex:screen          enable screen text output (on by default)
    --apex:quiet           disable screen text output
    --apex:csv             enable csv text output
    --apex:tau             enable tau profile output
    --apex:taskgraph       enable taskgraph output
                           (graphviz required for post-processing)
    --apex:otf2            enable OTF2 trace output
    --apex:otf2path        specify location of OTF2 archive
                           (default: ./OTF2_archive)
    --apex:otf2name        specify name of OTF2 file (default: APEX)
    --apex:gtrace          enable Google Trace Events output
    --apex:scatter         enable scatterplot output
                           (python required for post-processing)
    --apex:openacc         enable OpenACC support
    --apex:kokkos          enable Kokkos support
    --apex:raja            enable RAJA support
    --apex:pthread         enable pthread wrapper support
    --apex:untied          enable tasks to migrate cores/OS threads
                           during execution (not compatible with trace output)
    --apex:cuda_counters   enable CUDA/CUPTI counter support
    --apex:cuda_driver     enable CUDA driver API callbacks
    --apex:cuda_details    enable per-kernel statistics where available
    --apex:cpuinfo         enable sampling of /proc/cpuinfo (Linux only)
    --apex:meminfo         enable sampling of /proc/meminfo (Linux only)
    --apex:net             enable sampling of /proc/net/dev (Linux only)
    --apex:status          enable sampling of /proc/self/status (Linux only)
    --apex:io              enable sampling of /proc/self/io (Linux only)
    --apex:period          specify frequency of OS/HW sampling
    --apex:ompt_simple     only enable OpenMP Tools required events
    --apex:ompt_details    enable all OpenMP Tools events
```
