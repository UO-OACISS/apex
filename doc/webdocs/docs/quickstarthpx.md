# APEX Quickstart

## Installation

For detailed instructions and information on dependencies, see [build instructions](install.md#installation_with_hpx).

APEX is integrated into the [HPX runtime](https://hpx.stellar-group.org), and is integrated into the HPX build system.  To enable APEX measurement with HPX, enable the following CMake flags:

```
-DHPX_WITH_APEX=TRUE
```

The `-DHPX_WITH_APEX_TAG=develop` can be used to indicate a specific release version of APEX, or to use a specific GitHub branch of APEX.  We recommend using the default configured version that comes with HPX (currently `v2.3.1`) or the `develop` branch.

## Runtime

To see APEX data after an HPX run, set the `APEX_SCREEN_OUTPUT=1` environment variable.  After execution, you'll see output like this:

```
[khuck@eagle build]$ export APEX_SCREEN_OUTPUT=1
[khuck@eagle build]$ ./bin/fibonacci
fibonacci(10) == 55
elapsed time: 0.112029 [s]

Elapsed time: 0.19137 seconds
Cores detected: 128
Worker Threads observed: 32
Available CPU time: 6.12383 seconds

Timer                                                : #calls  |    mean  |   total  |  % total
------------------------------------------------------------------------------------------------
                                           APEX MAIN :        1      0.191      0.191    100.000
                                               async :        2      0.000      0.000      0.001
                        async_launch_policy_dispatch :        5      0.001      0.003      0.041
            broadcast_call_shutdown_functions_action :        2      0.000      0.001      0.012
                      call_shutdown_functions_action :        2      0.002      0.005      0.081
                                    fibonacci_action :      174      0.015      2.569     41.957
                              load_components_action :        1      0.014      0.014      0.230
                   primary_namespace_colocate_action :        2      0.000      0.001      0.011
                                          run_helper :        1      0.015      0.015      0.250
                                 shutdown_all_action :        1      0.002      0.002      0.040
                                           APEX Idle :                          3.514     57.375
------------------------------------------------------------------------------------------------
                                        Total timers : 190
```
