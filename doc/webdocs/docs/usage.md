
# Supported Runtime Systems

## HPX (Louisiana State University)

HPX (High Performance ParalleX) is the original implementation of the ParalleX model. Developed and maintained by the Ste||ar Group at Louisiana State University, HPX is implemented in C++. For more information, see <http://stellar.cct.lsu.edu/tag/hpx/>.  For a tutorial on HPX with APEX (presented at SC'15, Austin TX) see <https://github.com/khuck/SC15_APEX_tutorial>.

### Configuring HPX with APEX
Coming soon... For now, see <https://github.com/khuck/SC15_APEX_tutorial>.

## HPX-5 (Indiana University)

HPX-5 (High Performance ParalleX) is a second implementation of the ParalleX model. Developed and maintained by the CREST Group at Indiana University, HPX-5 is implemented in C.  For more information, see <https://hpx.crest.iu.edu>.

### Configuring HPX-5 with APEX

APEX is built as a pre-requisite dependency of HPX-5. So, before configuring and building HPX-5, configure and build APEX as a standalone library.  In addition to the usual required options for CMake, we will also include the options to specify Binutils support, because the HPX-5 instrumentation uses function addresses to identify timers rather than strings.  To include Binutils, we can choose one of:

* use a system-installed binutils by specifying `-DUSE_BFD=TRUE`
* use a custom build of Binutils by specifying `-DUSE_BFD=TRUE -DBFD_ROOT=<path-to-binutils-installation>
* have APEX download and build Binutils automatically by specifying `-DBUILD_BFD=TRUE`.

For example, assume Boost is installed in /usr/local/boost/1.54.0, we will have CMake download and build Binutils, and we want to install APEX to /usr/local/apex/0.1.  To configure, build and install APEX in the main source directory (your paths may vary):

```bash
cd $HOME/src
wget https://github.com/khuck/xpress-apex/archive/v0.1.tar.gz
tar -xvzf v0.1.tar.gz
cd xpress-apex-0.1
mkdir build
cd build
cmake -DBOOST_ROOT=/usr/local/boost/1.54.0 -DBUILD_BFD=TRUE -DCMAKE_INSTALL_PREFIX=/usr/local/xpress-apex/0.1 -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
make test # optional
make doc # optional
make install
```

Keep in mind that APEX will automatically download, configure and build Active Harmony as part of the build process, unless you pass `-DUSE_ACTIVEHARMONY=FALSE` to the cmake command.  After the build is complete, add the package configuration path to your PKG_CONFIG_PATH environment variable (HPX-5 uses autotools for configuration so it will find APEX using the utility pkg-config):

```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/xpress-apex/0.1/lib/pkgconfig
```

To confirm the PKG_CONFIG_PATH variable is set correctly, try executing the pkg-config command:

```bash
pkg-config --libs apex
```

Which should give the following output (or something similar):

```bash
-L/usr/local/xpress-apex/lib -L/usr/lib -lapex -lboost_system -lboost_thread -lboost_timer -lboost_chrono -lboost_regex -lpthread -lbfd -liberty -lz -lm -Wl,-rpath,/usr/local/xpress-apex/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu,-rpath,/usr/lib -lstdc++
```

Once APEX is installed, you can configure and build HPX-5 with APEX.  To include APEX in the HPX-5 configuration, include the --with-apex=yes option when calling configure.  Assuming you have downloaded HPX-5 v.2.0, you would do the following:

```bash
cd HPX_Release_v2.0.0/hpx
./bootstrap
./configure --enable-apps --enable-testsuite --prefix=/home/khuck/src/hpx-iu/hpx-install --with-apex=yes
make -j8
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

## OpenMP

The OpenMP API supports multi-platform shared-memory parallel programming in C/C++ and Fortran. The OpenMP API defines a portable, scalable model with a simple and flexible interface for developing parallel applications on platforms from the desktop to the supercomputer.  For more information, see <http://openmp.org/>.

### Configuring HPX with APEX
Coming soon... APEX comes with several OpenMP examples.

