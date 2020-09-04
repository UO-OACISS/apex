# APEX Quickstart

## Installation

For detailed instructions and information on dependencies, see [build instructions](install.md#standalone_installation)

To build APEX stand-alone (to use with OpenMP, OpenACC, CUDA, Kokkos, TBB, C++ threads, etc.) do the following:

```
git clone https://github.com/khuck/xpress-apex.git
cd xpress-apex
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=TRUE ..
make -j
```

## Runtime

To run an example (since `-DBUILD_EXAMPLES=TRUE` was set), just run the Matmult example and you should get similar output:

```
[khuck@eagle xpress-apex]$ ./build/src/examples/Matmult/matmult
Spawned thread 1...
Spawned thread 2...
Spawned thread 3...
Done.

Elapsed time: 0.300207 seconds
Cores detected: 128
Worker Threads observed: 4
Available CPU time: 1.20083 seconds

Counter                                   : #samples | minimum |    mean  |  maximum |  stddev
------------------------------------------------------------------------------------------------
                           status:Threads :        1      6.000      6.000      6.000      0.000
                            status:VmData :        1   4.93e+04   4.93e+04   4.93e+04      0.000
                             status:VmExe :        1     64.000     64.000     64.000      0.000
                             status:VmHWM :        1   7808.000   7808.000   7808.000      0.000
                             status:VmLck :        1      0.000      0.000      0.000      0.000
                             status:VmLib :        1   6336.000   6336.000   6336.000      0.000
                             status:VmPMD :        1     16.000     16.000     16.000      0.000
                             status:VmPTE :        1      4.000      4.000      4.000      0.000
                            status:VmPeak :        1   3.80e+05   3.80e+05   3.80e+05      0.000
                             status:VmPin :        1      0.000      0.000      0.000      0.000
                             status:VmRSS :        1   7808.000   7808.000   7808.000      0.000
                            status:VmSize :        1   3.15e+05   3.15e+05   3.15e+05      0.000
                             status:VmStk :        1    192.000    192.000    192.000      0.000
                            status:VmSwap :        1      0.000      0.000      0.000      0.000
        status:nonvoluntary_ctxt_switches :        1      0.000      0.000      0.000      0.000
           status:voluntary_ctxt_switches :        1     77.000     77.000     77.000      0.000
------------------------------------------------------------------------------------------------

Timer                                                : #calls  |    mean  |   total  |  % total
------------------------------------------------------------------------------------------------
                                           APEX MAIN :        1      0.300      0.300    100.000
                                      allocateMatrix :       12      0.009      0.108      9.023
                                             compute :        4      0.206      0.825     68.736
                                 compute_interchange :        4      0.064      0.257     21.369
                                             do_work :        4      0.298      1.193     99.313
                                          freeMatrix :       12      0.000      0.000      0.025
                                          initialize :       12      0.000      0.002      0.146
                                                main :        1      0.299      0.299     24.930
------------------------------------------------------------------------------------------------
                                        Total timers : 49

```