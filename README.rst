*NB: A fairly recent version of the documentation is available here:*
http://www.nic.uoregon.edu/~khuck/apex_docs/doc/html/index.html

Overview:
=========

One of the key components of the XPRESS project is a new approach to performance observation, measurement, analysis and runtime decision making in order to optimize performance. The particular challenges of accurately measuring the performance characteristics of ParalleX applications requires a new approach to parallel performance observation. The standard model of multiple operating system processes and threads observing themselves in a first-person manner while writing out performance profiles or traces for offline analysis will not adequately capture the full execution context, nor provide opportunities for runtime adaptation within OpenX. The approach taken in the XPRESS project is a new performance measurement system, called (Autonomic Performance Environment for eXascale). APEX will include methods for information sharing between the layers of the software stack, from the hardware through operating and runtime systems, all the way to domain specific or legacy applications. The performance measurement components will incorporate relevant information across stack layers, with merging of third-person performance observation of node-level and global resources, remote processes, and both operating and runtime system threads.

Option 1: Configuring and building APEX with bootstrap scripts (deprecated):
===============================================================

copy and modify ./bootstrap-configs/bootstrap-$arch.sh as necessary, and run it.

Option 2: Configuring and building APEX with CMake directly (recommended):
============================================================

APEX is built with CMake. The minimum CMake settings needed for APEX are:

* -DBOOST_ROOT=<some path to the Boost installation>
* -DCMAKE_INSTALL_PREFIX=<some path to an installation location>
* -DCMAKE_BUILD_TYPE=<one of Release, Debug, or RelWithDebInfo (recommended)>

The process for building APEX is:

1) Get the code::

    wget https://github.com/khuck/xpress-apex/archive/v0.1.tar.gz
    tar -xvzf v0.1.tar.gz

2) Enter the repo directory, make a build directory::

    cd xpress-apex-0.1
    mkdir build
    cd build

3) configure using CMake::

    cmake -DBOOST_ROOT=<path-to-boost> -DCMAKE_INSTALL_PREFIX=<installation-path> -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

4) build with make::

    make
    make test
    make doc
    make install

other CMake settings, depending on your needs/wants:
----------------------------------------------------

* -DUSE_ACTIVEHARMONY=TRUE or FALSE
* -DACTIVEHARMONY_ROOT=some path to ActiveHarmony, or set the ACTIVEHARMONY_ROOT environment variable.
  It should be noted that if ActiveHarmony is not specified and -DUSE_ACTIVEHARMONY is TRUE or not set, APEX
  will download and build ActiveHarmony as a CMake project. To disable ActiveHarmony entirely, specify
  -DUSE_ACTIVEHARMONY=FALSE.

* -DUSE_OMPT=TRUE or FALSE
* -DOMPT_ROOT=path to OMPT, or set the OMPT_ROOT environment variable.

* -DUSE_BFD=TRUE or FALSE
* -DBFD_ROOT=path to Binutils, or set the BFD_ROOT environment variable.

* -DUSE_TAU=TRUE or FALSE
* -DTAU_ROOT=path to TAU, or set the TAU_ROOT environment variable.
* -DTAU_ARCH=the TAU architecture, like x86_64, craycnl, mic_linux, bgq, etc.
* -DTAU_OPTIONS=a TAU configuration with thread support, like "-pthread" or "-icpc-pthread"

* -DUSE_RCR=TRUE or FALSE
* -DRCR_ROOT=path to RCR, or set the RCR_ROOT environment variable.

* -DUSE_TCMALLOC=TRUE or FALSE
* -DTCMALLOC_ROOT=path to TCMalloc, or set the TCMALLOC_ROOT environment variable.

* -DUSE_JEMALLOC=TRUE or FALSE
* -DJEMALLOC_ROOT=path to JEMalloc, or set the JEMALLOC_ROOT environment variable.

* -DUSE_PAPI=TRUE or FALSE
* -DPAPI_ROOT=some path to PAPI, or set the PAPI_ROOT environment variable.

* -DUSE_LM_SENSORS=TRUE or FALSE

* -DBUILD_EXAMPLES=TRUE or FALSE
* -DBUILD_TESTS=TRUE or FALSE

* -DUSE_MPI=TRUE or FALSE (whether to build MPI global support/examples)
* -DMPI_C_INCLUDE_PATH=path to MPI headers
* -DMPI_CXX_INCLUDE_PATH=path to MPI headers
* -DMPI_C_LIBRARIES=paths to MPI libraries, library names
* -DMPI_CXX_LIBRARIES=paths to MPI libraries, library names
* -DMPI_C_COMPILER=mpicc
* -DMPI_CXX_COMPILER=mpicxx

* -DCMAKE_C_COMPILER=gcc
* -DCMAKE_CXX_COMPILER=g++
* -DCMAKE_BUILD_TYPE=Release, Debug, RelWithDebInfo
* -DBUILD_SHARED_LIBS=TRUE or FALSE
