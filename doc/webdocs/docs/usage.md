# Supported Runtime Systems

## HPX (Louisiana State University)

HPX (High Performance ParalleX) is the original implementation of the ParalleX model. Developed and maintained by the Ste||ar Group at Louisiana State University, HPX is implemented in C++. For more information, see <http://stellar.cct.lsu.edu/tag/hpx/>.  For a tutorial on HPX with APEX (presented at SC'15, Austin TX) see <https://github.com/khuck/SC15_APEX_tutorial>.

APEX is configured and built as part of HPX. In fact, you don't even need to donwload it separately - it will be automatically checked out from Github as part of the HPX Cmake configuration.  However, you do need to pass the correct Cmake options to the HPX configuration step.

### Using TAU to profile or trace HPX

If you want to use TAU to collect profiles or traces of HPX applications, you *will* need to download and configure TAU first.  The following instructions will include the TAU options, so please see [APEX with TAU](usecases.md#with-tau) for instructions on configuring TAU for use with APEX.  For this example, we assume TAU was configured with "./configure -mpi -pthread" on an x86_64 Linux machine. 

### Configuring HPX with APEX

After TAU is configured and built, we are ready to configure and build HPX. Configure HPX as usual, but add the following options:

```bash
mkdir build
cd build

# The source code is located in $HOME/src/hpx
cmake \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_INSTALL_PREFIX=. \
-DHPX_WITH_PARCELPORT_MPI=TRUE \
-DHPX_WITH_APEX=TRUE \
-DAPEX_WITH_ACTIVEHARMONY=TRUE \
-DHPX_WITH_TAU=TRUE -DTAU_ROOT=$HOME/src/tau2 \
-DTAU_ARCH=x86_64 -DTAU_OPTIONS=-mpi-pthread \
-DHPX_WITH_MALLOC=jemalloc \
$HOME/src/hpx

make -j5 core
make -j5 components
make -j3 examples
```

To confirm that HPX was configured and built with APEX correctly, run the simple HPX example:

```bash
export APEX_SCREEN_OUTPUT=1
./build/bin/fibonacci
```

Which should give output similar to this:

```bash
parquet-7a70deb-release
Built on: 12:06:08 Jan  5 2016
C++ Language Standard version : 201402
GCC Compiler version : 5.2.1 20151010
APEX_TAU : 0
APEX_POLICY : 1
APEX_MEASURE_CONCURRENCY : 0
APEX_MEASURE_CONCURRENCY_PERIOD : 1000000
APEX_SCREEN_OUTPUT : 1
APEX_PROFILE_OUTPUT : 0
APEX_CSV_OUTPUT : 0
APEX_TASKGRAPH_OUTPUT : 0
APEX_PROC_CPUINFO : 0
APEX_PROC_MEMINFO : 0
APEX_PROC_NET_DEV : 0
APEX_PROC_SELF_STATUS : 0
APEX_PROC_STAT : 1
APEX_THROTTLE_CONCURRENCY : 0
APEX_THROTTLING_MAX_THREADS : 8
APEX_THROTTLING_MIN_THREADS : 1
APEX_THROTTLE_ENERGY : 0
APEX_THROTTLING_MAX_WATTS : 300
APEX_THROTTLING_MIN_WATTS : 150
APEX_PTHREAD_WRAPPER_STACK_SIZE : 0
APEX_PAPI_METRICS : 
fibonacci(10) == 55
elapsed time: 0.00595707 [s]
Info: 1 items remaining on on the profiler_listener queue...done.
CPU is 2.66024e+09 Hz.
Elapsed time: 0.122906
Cores detected: 8
Worker Threads observed: 1
Available CPU time: 0.122906
Action                         :  #calls  |  minimum |    mean  |  maximum |   total  |  stddev  |  % total  
------------------------------------------------------------------------------------------------------------
              APEX MAIN THREAD :        1    --n/a--   1.22e-01    --n/a--   1.22e-01    --n/a--     99.631
broadcast_call_shutdown_fun... :        2    --n/a--   2.59e-05    --n/a--   5.18e-05    --n/a--      0.042
broadcast_call_startup_func... :        2    --n/a--   2.36e-05    --n/a--   4.72e-05    --n/a--      0.038
call_shutdown_functions_action :        2    --n/a--   2.29e-04    --n/a--   4.58e-04    --n/a--      0.373
 call_startup_functions_action :        2    --n/a--   8.01e-05    --n/a--   1.60e-04    --n/a--      0.130
              fibonacci_action :      177    --n/a--   1.28e-05    --n/a--   2.26e-03    --n/a--      1.841
                      hpx_main :        1    --n/a--   4.13e-04    --n/a--   4.13e-04    --n/a--      0.336
        load_components_action :        2    --n/a--   1.96e-03    --n/a--   3.92e-03    --n/a--      3.192
                      pre_main :        1    --n/a--   1.39e-03    --n/a--   1.39e-03    --n/a--      1.135
primary_namespace_bulk_serv... :       24    --n/a--   2.35e-05    --n/a--   5.65e-04    --n/a--      0.459
primary_namespace_service_a... :        4    --n/a--   1.02e-05    --n/a--   4.08e-05    --n/a--      0.033
                    run_helper :        1    --n/a--   2.80e-04    --n/a--   2.80e-04    --n/a--      0.228
symbol_namespace_service_ac... :        4    --n/a--   1.73e-05    --n/a--   6.93e-05    --n/a--      0.056
                     APEX Idle :  --n/a--    --n/a--    --n/a--    --n/a--    --n/a--    --n/a--    --n/a--   
------------------------------------------------------------------------------------------------------------
```

To enable TAU profiling, set the APEX_TAU environment variable to 1.  We will also set some other TAU environment varaibles and re-run the program:

```bash
export APEX_TAU=1
export TAU_PROFILE_FORMAT=merged
./build/bin/fibonacci
```

The "merged" profile setting will create a single file (tauprofile.xml) for the whole application, rather than a profile.* file for each thread.

After execution, there is a TAU profile file called "tauprofile.xml".  To view the results of the profiling, run the ParaProf application on the profile (assuming the TAU utilities are in your path):

```bash
paraprof tauprofile.xml
```

For more information on using TAU with APEX, see [APEX with TAU](usecases.md#with-tau).

## HPX-5 (Indiana University)

HPX-5 (High Performance ParalleX) is a second implementation of the ParalleX model. Developed and maintained by the CREST Group at Indiana University, HPX-5 is implemented in C.  For more information, see <https://hpx.crest.iu.edu>.

### Configuring HPX-5 with APEX

APEX is built as a pre-requisite dependency of HPX-5. So, before configuring and building HPX-5, configure and build APEX as a standalone library.  In addition to the usual required options for CMake, we will also include the options to include Active Harmony (for policies), TAU (for performance analysis - see [APEX with TAU](usecases.md#with-tau) for instructions on configuring TAU) and Binutils support, because the HPX-5 instrumentation uses function addresses to identify timers rather than strings.  To include Binutils, we can choose one of:

* use a system-installed binutils by specifying `-DUSE_BFD=TRUE`
* use a custom build of Binutils by specifying `-DUSE_BFD=TRUE -DBFD_ROOT=<path-to-binutils-installation>`
* have APEX download and build Binutils automatically by specifying `-DBUILD_BFD=TRUE`.

**Note:** *HPX-5 uses JEMalloc, TBB Malloc or DLMalloc, so **DO NOT** configure APEX with either TCMalloc or JEMalloc.*

For example, assume TAU is installed in /usr/local/tau/2.25 and we will have CMake download and build Binutils and Active Harmony, and we want to install APEX to /usr/local/apex/0.5.  To configure, build and install APEX in the main source directory (your paths may vary):

```bash
cd $HOME/src
wget https://github.com/khuck/xpress-apex/archive/v0.5.tar.gz
tar -xvzf v0.5.tar.gz
cd xpress-apex-0.5
mkdir build
cd build
cmake \
-DUSE_TAU=TRUE -DTAU_ROOT=/usr/local/tau/2.25 -DTAU_OPTIONS=-papi-pthread -DTAU_ARCH=x86_64 \
-DBUILD_BFD=TRUE -DCMAKE_INSTALL_PREFIX=/usr/local/xpress-apex/0.5 -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
make test # optional
make doc # optional
make install
```

Keep in mind that APEX will automatically download, configure and build Active Harmony as part of the build process, unless you pass `-DUSE_ACTIVEHARMONY=FALSE` to the cmake command.  After the build is complete, add the package configuration path to your PKG_CONFIG_PATH environment variable (HPX-5 uses autotools for configuration so it will find APEX using the utility pkg-config):

```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/xpress-apex/0.5/lib/pkgconfig
```

To confirm the PKG_CONFIG_PATH variable is set correctly, try executing the pkg-config command:

```bash
pkg-config --libs apex
```

Which should give the following output (or something similar):

```bash
-L/usr/local/xpress-apex/0.5/lib -L/usr/local/tau/2.25/x86_64/lib -L/usr/local/xpress-apex/0.5/lib -lapex -lpthread -lTAUsh-papi-pthread -lharmony -lbfd -liberty -lz -lm -Wl,-rpath,/usr/local/tau/2.25/x86_64/lib,-rpath,/usr/local/xpress-apex/0.5/lib -lstdc++
```

Once APEX is installed, you can configure and build HPX-5 with APEX.  To include APEX in the HPX-5 configuration, include the --with-apex=yes option when calling configure.  Assuming you have downloaded HPX-5 v.3.0, you would do the following:

```bash
# go to the HPX source directory
cd HPX_Release_v3.0.0/hpx
# If you haven't already set the pkgconfig path, do so now...
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/xpress-apex/0.5/lib/pkgconfig
# configure
./bootstrap
./configure --enable-testsuite --prefix=/home/khuck/src/hpx-iu/hpx-install --with-apex=yes
# build!
make -j8
# install!
make install
```

To confirm that HPX-5 was configured and built with APEX correctly, run the simple APEX example:

```bash
export APEX_SCREEN_OUTPUT=1
./tests/unit/apex
```

Which should give output similar to this:

```bash
v0.1-5e4ac87-master
Built on: 13:23:34 Dec 17 2015
C++ Language Standard version : 201402
GCC Compiler version : 5.2.1 20151010
APEX_TAU : 0
APEX_POLICY : 1
APEX_MEASURE_CONCURRENCY : 0
APEX_MEASURE_CONCURRENCY_PERIOD : 1000000
APEX_SCREEN_OUTPUT : 1
APEX_PROFILE_OUTPUT : 0
APEX_CSV_OUTPUT : 0
APEX_TASKGRAPH_OUTPUT : 0
APEX_PROC_CPUINFO : 0
APEX_PROC_MEMINFO : 0
APEX_PROC_NET_DEV : 0
APEX_PROC_SELF_STATUS : 0
APEX_PROC_STAT : 1
APEX_THROTTLE_CONCURRENCY : 0
APEX_THROTTLING_MAX_THREADS : 8
APEX_THROTTLING_MIN_THREADS : 1
APEX_THROTTLE_ENERGY : 0
APEX_THROTTLING_MAX_WATTS : 300
APEX_THROTTLING_MIN_WATTS : 150
APEX_PTHREAD_WRAPPER_STACK_SIZE : 0
APEX_PAPI_METRICS : 

Missing fib number. Using 10.
fib(10)=55
seconds: 0.0005629
localities: 1
threads/locality: 8
Info: 34 items remaining on on the profiler_listener queue...done.
CPU is 2.66036e+09 Hz.
Elapsed time: 0.0364015
Cores detected: 8
Worker Threads observed: 8
Available CPU time: 0.291212
Action                         :  #calls  |  minimum |    mean  |  maximum |   total  |  stddev  |  % total  
------------------------------------------------------------------------------------------------------------
_fib_main_action [{/home/kh... :        1    --n/a--   4.52e-04    --n/a--   4.52e-04    --n/a--      0.155
_fib_action [{/home/khuck/s... :      177    --n/a--   4.39e-06    --n/a--   7.77e-04    --n/a--      0.267
_locality_stop_handler [{/h... :        1    --n/a--   1.21e-05    --n/a--   1.21e-05    --n/a--      0.004
                 failed steals :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                          mail :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        spawns :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        stacks :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        steals :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        yields :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                     APEX Idle :  --n/a--    --n/a--    --n/a--    --n/a--   2.90e-01    --n/a--     99.574
------------------------------------------------------------------------------------------------------------
```
### Building HPX-5 applications with APEX

APEX will automatically be included in the link when HPX-5 applciations are built. To build an example, go to the hpx-apps directory and build the LULESH parcels example:

```bash
cd hpx-apps/lulesh/parcels
# assuming HPX-5 is installed in /usr/local/hpx/3.0, set the pkgconfig path
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/hpx/3.0/lib/pkgconfig
# configure 
./bootstrap
./configure
# make!
make
```

Then, to run the LULESH example:

```bash
export APEX_SCREEN_OUTPUT=1
./luleshparcels -n 8 -x 24 -i 100 --hpx-threads=8
```

Should give the following output (or similar):

```bash
v0.1-907c977-master
Built on: 09:50:08 Dec 23 2015
C++ Language Standard version : 201402
GCC Compiler version : 5.2.1 20151010
APEX_TAU : 0
APEX_POLICY : 1
APEX_MEASURE_CONCURRENCY : 0
APEX_MEASURE_CONCURRENCY_PERIOD : 1000000
APEX_SCREEN_OUTPUT : 1
APEX_PROFILE_OUTPUT : 0
APEX_CSV_OUTPUT : 0
APEX_TASKGRAPH_OUTPUT : 0
APEX_PROC_CPUINFO : 0
APEX_PROC_MEMINFO : 0
APEX_PROC_NET_DEV : 0
APEX_PROC_SELF_STATUS : 0
APEX_PROC_STAT : 1
APEX_THROTTLE_CONCURRENCY : 0
APEX_THROTTLING_MAX_THREADS : 8
APEX_THROTTLING_MIN_THREADS : 1
APEX_THROTTLE_ENERGY : 0
APEX_THROTTLING_MAX_WATTS : 300
APEX_THROTTLING_MIN_WATTS : 150
APEX_PTHREAD_WRAPPER_STACK_SIZE : 0
APEX_PAPI_METRICS : 
 Number of domains: 8 nx: 24 maxcycles: 100 core-major ordering: 1


START_LOG
PROGNAME: lulesh-parcels

Elapsed time = 1.255209e+01

Run completed:  
  Problem size = 24 
  Iteration count = 100 
  Final Origin Energy = 4.739209e+06
  Testing plane 0 of energy array:
  MaxAbsDiff   = 9.313226e-10
  TotalAbsDiff = 2.841568e-09
  MaxRelDiff   = 2.946213e-12

END_LOG

time_in_SBN3 = 4.570989e-01
time_in_PosVel = 2.182410e-01
time_in_MonoQ = 4.889381e+00
 Elapsed: 12599.4
CPU is 2.66028e+09 Hz.
Elapsed time: 12.6192
Cores detected: 8
Worker Threads observed: 8
Available CPU time: 100.953
Action                         :  #calls  |  minimum |    mean  |  maximum |   total  |  stddev  |  % total  
------------------------------------------------------------------------------------------------------------
_advanceDomain_action [{/ho... :        8    --n/a--   1.17e+01    --n/a--   9.34e+01    --n/a--     92.506
_initDomain_action [{/home/... :        8    --n/a--   2.04e-02    --n/a--   1.63e-01    --n/a--      0.162
_finiDomain_action [{/home/... :        8    --n/a--   2.81e-03    --n/a--   2.25e-02    --n/a--      0.022
_main_action [{/home/khuck/... :        1    --n/a--   4.73e-03    --n/a--   4.73e-03    --n/a--      0.005
_SBN1_result_action [{/home... :       56    --n/a--   1.42e-03    --n/a--   7.93e-02    --n/a--      0.079
_SBN1_sends_action [{/home/... :       56    --n/a--   1.87e-04    --n/a--   1.05e-02    --n/a--      0.010
_SBN3_result_action [{/home... :     5600    --n/a--   1.33e-04    --n/a--   7.45e-01    --n/a--      0.738
_SBN3_sends_action [{/home/... :     5600    --n/a--   9.05e-05    --n/a--   5.07e-01    --n/a--      0.502
_PosVel_result_action [{/ho... :     2800    --n/a--   1.61e-04    --n/a--   4.50e-01    --n/a--      0.445
_PosVel_sends_action [{/hom... :     2800    --n/a--   1.43e-04    --n/a--   4.00e-01    --n/a--      0.396
_MonoQ_result_action [{/hom... :     2400    --n/a--   1.03e-04    --n/a--   2.47e-01    --n/a--      0.245
_MonoQ_sends_action [{/home... :     2400    --n/a--   1.79e-04    --n/a--   4.29e-01    --n/a--      0.425
_locality_stop_handler [{/h... :        1    --n/a--   2.45e-04    --n/a--   2.45e-04    --n/a--      0.000
_allreduce_init_handler [{/... :        2    --n/a--   5.49e-04    --n/a--   1.10e-03    --n/a--      0.001
_allreduce_fini_handler [{/... :        2    --n/a--   2.44e-04    --n/a--   4.89e-04    --n/a--      0.000
_allreduce_add_handler [{/h... :        9    --n/a--   6.74e-05    --n/a--   6.07e-04    --n/a--      0.001
_allreduce_remove_handler [... :        9    --n/a--   4.31e-05    --n/a--   3.88e-04    --n/a--      0.000
_allreduce_join_handler [{/... :       99    --n/a--   4.90e-05    --n/a--   4.86e-03    --n/a--      0.005
_allreduce_bcast_handler [{... :       99    --n/a--   2.75e-05    --n/a--   2.72e-03    --n/a--      0.003
                   CPU Guest % :       12      0.000      0.000      0.000      0.000      0.000    --n/a-- 
                CPU I/O Wait % :       12      0.000      0.000      0.000      0.000      0.000    --n/a-- 
                     CPU IRQ % :       12      0.000      0.000      0.000      0.000      0.000    --n/a-- 
                    CPU Idle % :       12      0.000      0.789      8.429      9.464      2.305    --n/a-- 
                    CPU Nice % :       12      0.000      0.000      0.000      0.000      0.000    --n/a-- 
                   CPU Steal % :       12      0.000      0.000      0.000      0.000      0.000    --n/a-- 
                  CPU System % :       12     21.000     22.387     24.286    268.643      0.941    --n/a-- 
                    CPU User % :       12     77.500     80.426     89.714    965.107      4.315    --n/a-- 
                CPU soft IRQ % :       12      0.000      0.010      0.125      0.125      0.035    --n/a-- 
                 failed steals :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                          mail :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        spawns :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        stacks :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        steals :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                        yields :        1   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00    --n/a-- 
                     APEX Idle :  --n/a--    --n/a--    --n/a--    --n/a--   4.50e+00    --n/a--      4.455
------------------------------------------------------------------------------------------------------------
```

To enable TAU profiling, set the APEX_TAU environment variable to 1.  We will also set some other TAU environment varaibles and re-run the program:

```bash
export APEX_TAU=1
export TAU_PROFILE_FORMAT=merged
export TAU_SAMPLING=1
./luleshparcels -n 8 -x 24 -i 100 --hpx-threads=8
```

The "merged" profile setting will create a single file (tauprofile.xml) for the whole application, rather than a profile.* file for each thread. The sampling flag will enable periodic interruption of the application to get a more detailed profile.

After execution, there is a TAU profile file called "tauprofile.xml".  To view the results of the profiling, run the ParaProf application on the profile (assuming the TAU utilities are in your path):

```bash
paraprof tauprofile.xml
```

Which should result in a profile like the following:

![Screenshot](img/lulesh-paraprof1.tiff)

*Above: ParaProf main profiler window showing all threads of execution.*

![Screenshot](img/lulesh-paraprof2.tiff)

*Above: ParaProf main profiler window showing one thread of execution.*

![Screenshot](img/lulesh-paraprof3.tiff)

*Above: ParaProf main profiler window showing one thread of execution, in a callgraph view.*

For more information on using TAU with APEX, see [APEX with TAU](usecases.md#with-tau).


## OpenMP

The OpenMP API supports multi-platform shared-memory parallel programming in C/C++ and Fortran. The OpenMP API defines a portable, scalable model with a simple and flexible interface for developing parallel applications on platforms from the desktop to the supercomputer.  For more information, see <http://openmp.org/>.

### Running OpenMP applications with APEX
*...Coming soon!*

