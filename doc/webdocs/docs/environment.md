There are a number of environment variables that control APEX behavior
at runtime. The variables can be defined in the environment before 
application execution, or specified in a file called *apex.conf* in the
current execution directory.

| Environment Variable | Default Value | Valid Values | Description |
| -------------------- | -- | -- | -------------------------------- |
| APEX_SCREEN_OUTPUT | 1 | 0,1 | Output APEX options at entry, and performance summary at exit |
| APEX_PROFILE_OUTPUT | 0 | 0,1 | Output TAU profile of performance summary |
| APEX_CSV_OUTPUT | 0 | 0,1 | Output CSV profile of performance summary |
| APEX_TASKGRAPH_OUTPUT | 0 | 0,1 | Output graphviz reduced taskgraph |
| APEX_POLICY | 1 | 0,1 | Enable APEX policy listener and execute registered policies |
| APEX_PROC_STAT | 1 | 0,1 | Periodically read data from /proc/stat |
| APEX_PROC_CPUINFO | 0 | 0,1 | Read data (once) from /proc/cpuinfo |
| APEX_PROC_MEMINFO | 0 | 0,1 | Periodically read data from /proc/meminfo |
| APEX_PROC_NET_DEV | 0 | 0,1 | Periodically read data from /proc/net/dev |
| APEX_PROC_SELF_STATUS | 0 | 0,1 | Periodically read data from /proc/self/status |
| APEX_MEASURE_CONCURRENCY | 0 | 0,1 | Periodically sample thread activity and output report at exit |
| APEX_MEASURE_CONCURRENCY_PERIOD | 1000000 | Integer | Thread concurrency sampling period, in microseconds |
| APEX_TAU | 0 | 0,1 | Enable TAU profiling (if APEX is configured with TAU). |
| APEX_THROTTLE_CONCURRENCY | 0 | 0,1 | Deactivate/activate threads for policy support |
| APEX_THROTTLING_MIN_THREADS | 1 | 0,1 | Minimum threads allowed |
| APEX_THROTTLING_MAX_THREADS | 8 | 0,1 | Maximum threads allowed |
| APEX_THROTTLE_ENERGY | 0 | 0,1 | Enable energy throttling |
| APEX_THROTTLING_MIN_WATTS | 150 | Integer | Minimum Watt threshold |
| APEX_THROTTLING_MAX_WATTS | 300 | Integer | Maximum Watt threshold |
| APEX_PTHREAD_WRAPPER_STACK_SIZE | 0 | 16k-8M | When wrapping pthread_create, use this size for the stack. |
| APEX_PAPI_METRICS | *null* | space-delimited string of metric names | List of metrics to be measured by APEX when timers are used. Only meaningful if APEX is configured with PAPI support.  Any supported metric from *papi_avail* ([see PAPI Documentation](http://icl.cs.utk.edu/projects/papi/wiki/PAPIC:papi_avail.1)) can be used. |

