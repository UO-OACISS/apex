.. contents:: Table of Contents

.. NOTE:: A fairly recent version of the API reference documentation is available here: <http://www.nic.uoregon.edu/~khuck/apex_docs/doc/html/index.html>


Overview:
=========

One of the key components of the XPRESS project is a new approach to performance observation, measurement, analysis and runtime decision making in order to optimize performance. The particular challenges of accurately measuring the performance characteristics of ParalleX applications (as well as other asynchronous multitasking runtime architectures) requires a new approach to parallel performance observation. The standard model of multiple operating system processes and threads observing themselves in a first-person manner while writing out performance profiles or traces for offline analysis will not adequately capture the full execution context, nor provide opportunities for runtime adaptation within OpenX. The approach taken in the XPRESS project is a new performance measurement system, called (Autonomic Performance Environment for eXascale). APEX includes methods for information sharing between the layers of the software stack, from the hardware through operating and runtime systems, all the way to domain specific or legacy applications. The performance measurement components will incorporate relevant information across stack layers, with merging of third-person performance observation of node-level and global resources, remote processes, and both operating and runtime system threads.  For a complete description of APEX, see the publication "APEX: An Autonomic Performance Environment for eXascale" [#]_.

Introspection
-------------
APEX provides an API for measuring actions within a runtime. The API includes methods for timer start/stop, as well as sampled counter values. APEX is designed to be integrated into a runtime, library or application and provide performance introspection for the purpose of runtime adaptation. While APEX can provide rudimentary post-mortem performance analysis measurement, there are many other performance measurement tools that perform that task much better (such as TAU <http://tau.uoregon.edu>).  That said, APEX includes an event listener that integrates with the TAU measurement system, so APEX events can be collected in a TAU profile and/or trace, and used for post-mortem performance anlaysis.

Runtime Adaptation
------------------
APEX provides a mechanism for dynamic runtime behavior, either for autotuning or adaptation to changing environment.  The infrastruture that provides the adaptation is the Policy Engine, which executes policies either periodically or triggered by events. The policies have access to the performance state as observed by the APEX introspection API. APEX is integrated with Active Harmony (<http://www.dyninst.org/harmony>) to provide dynamic search for autotuning.

Installation
============

Option 1: Configuring and building APEX with bootstrap scripts (deprecated, but useful for 'exotic' HPC machines):
----

copy and modify ./bootstrap-configs/bootstrap-$arch.sh as necessary, and run it.

Option 2: Configuring and building APEX with CMake directly (recommended):
----

APEX is built with CMake. The minimum CMake settings needed for APEX are:

* `-DBOOST_ROOT=<the path to a Boost installation, 1.54 or newer>`
* `-DCMAKE_INSTALL_PREFIX=<some path to an installation location>`
* `-DCMAKE_BUILD_TYPE=<one of Release, Debug, or RelWithDebInfo (recommended)>`

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
There are several utility libraries that provide functionality in APEX. Not all libraries are required, but some are recommended.  For the following options, the default values are in *italics*.

* **-DBUILD\_BOOST**=TRUE or *FALSE*.  In the event that Boost isn't pre-installed on your system, this option forces CMake to download and build Boost as part of the APEX project.
* **-DUSE\_ACTIVEHARMONY**=*TRUE* or FALSE.  Active Harmony is a library that intelligently searches for parametric combinations to support adapting to heterogeneous and changing environments.  For more information, see <http://www.dyninst.org/harmony>.  APEX uses Active Harmony for runtime adaptation.
* **-DACTIVEHARMONY\_ROOT**=the path to Active Harmony, or set the `ACTIVEHARMONY_ROOT` environment variable before running cmake.  It should be noted that if Active Harmony is not specified and `-DUSE_ACTIVEHARMONY` is TRUE or not set, APEX will download and build Active Harmony as a CMake project. To disable Active Harmony entirely, specify `-DUSE_ACTIVEHARMONY=FALSE`.
* **-DBUILD\_ACTIVEHARMONY**=TRUE or *FALSE*.  Whether or not Active Harmony is installed on the system, this option forces CMake to automatically download and build Active Harmony as part of the APEX project.
* **-DUSE\_OMPT**=TRUE or *FALSE*.  OMPT is a proposed standard for OpenMP runtimes to provide callback hooks to performance tools. For more information, see <http://openmp.org/mp-documents/ompt-tr2.pdf>.  APEX has support for most OMPT OpenMP trace events.
* **-DOMPT\_ROOT**=the path to OMPT, or set the `OMPT_ROOT` environment variable before running cmake.
* **-DBUILD\_OMPT**=TRUE or *FALSE*. Whether or not an OpenMP library with OMPT support is found by CMake, this option forces CMake to automatically download and build an OpenMP runtime with OMPT support as part of the APEX project.
* **-DUSE\_BFD**=TRUE or *FALSE*.  APEX uses libbfd to convert instruction addresses to source code locations. BFD support is useful for generating human-readable output for summaries and concurrency graphs. Libbfd is not required for runtime adaptation.
* **-DBFD\_ROOT**=path to Binutils, or set the `BFD_ROOT` environment variable.
* **-DBUILD\_BFD**=TRUE or FALSE.  Whether or not binutils is found by CMake, this option forces CMake to automatically download and build binutils as part of the APEX project.
* **-DUSE\_TAU**=TRUE or *FALSE*.  TAU (Tuning and Analysis Utilities) is a performance measurement and analysis framework for large scale parallel applications. For more information see <http://tau.uoregon.edu>.  APEX uses TAU to generate profiles for post-mortem performance analysis.
* **-DTAU\_ROOT**=path to TAU, or set the `TAU_ROOT` environment variable before running cmake.
* **-DTAU\_ARCH**=the TAU architecture, like `x86_64`, `craycnl`, `mic_linux`, `bgq`, etc.
* **-DTAU\_OPTIONS**=a TAU configuration with thread support, like `-pthread` or `-icpc-pthread`.
* **-DUSE\_RCR**=TRUE or *FALSE*.  RCR (Resource Centric Reflection) is a library for system monitoring of resources that require root access.  For more information, see <http://www.renci.org/wp-content/pub/techreports/TR-10-01.pdf>.  APEX uses RCR to access 'uncore' counters and system health information such as power and energy counters.
* **-DRCR\_ROOT**=the path to RCR, or set the `RCR_ROOT` environment variable.
* **-DUSE\_TCMALLOC**=TRUE or *FALSE*.  TCMalloc is a heap management library distributed as part of Google perftools. For more information, see <https://github.com/gperftools/gperftools>.  TCMalloc provides faster memory performance in multithreaded environments.
* **-DTCMALLOC\_ROOT**=path to TCMalloc, or set the `TCMALLOC_ROOT` environment variable before running cmake.
* **-DUSE\_JEMALLOC**=TRUE or `FALSE`.  JEMalloc is a heap management library.  For more information, see <http://www.canonware.com/jemalloc/>.  JEMalloc provides faster memory performance in multithreaded environments.
* **-DJEMALLOC\_ROOT**=path to JEMalloc, or set the `JEMALLOC_ROOT` environment variable before running cmake.
* **-DUSE\_PAPI**=TRUE or *FALSE*.  PAPI (Performance Application Programming Interface) provides the tool designer and application engineer with a consistent interface and methodology for use of the performance counter hardware found in most major microprocessors.  For more information, see <http://icl.cs.utk.edu/papi/>.  APEX uses PAPI to optionally collect hardware counters for timed events.
* **-DPAPI\_ROOT**=some path to PAPI, or set the `PAPI_ROOT` environment variable before running cmake.
* **-DUSE\_LM\_SENSORS**=TRUE or *FALSE*. Lm\_sensors (Linux Monitoring Sensors) is a library for monitoring hardware temperatures and fan speeds. For more information, see <https://en.wikipedia.org/wiki/Lm_sensors>.  APEX uses lm\_sensors to monitor hardware, where available.
* **-DBUILD\_EXAMPLES**=TRUE or *FALSE*. Whether or not to build the application examples in APEX.
* **-DBUILD\_TESTS**=*TRUE* or FALSE. Whether or not to build the APEX unit tests.
* **-DCMAKE\_C\_COMPILER**=*gcc*
* **-DCMAKE\_CXX\_COMPILER**=*g++*
* **-DCMAKE\_BUILD\_TYPE**=Release, *Debug*, RelWithDebInfo. Unfortunately, the cmake default (when not specified) is Debug. For faster performance, configure APEX to build `RelWithDebInfo` or `Release`.
* **-DBUILD\_SHARED\_LIBS**=TRUE or FALSE
* **-DUSE\_MPI**=TRUE or *FALSE*. Whether to build MPI global support and related examples.
* **-DMPI\_C\_INCLUDE\_PATH**=path to MPI headers
* **-DMPI\_CXX\_INCLUDE\_PATH**=path to MPI headers
* **-DMPI\_C\_LIBRARIES**=paths to MPI libraries, library names
* **-DMPI\_CXX\_LIBRARIES**=paths to MPI libraries, library names
* **-DMPI\_C\_COMPILER**=mpicc
* **-DMPI\_CXX\_COMPILER**=mpicxx

References
==========
    .. [#] Kevin A. Huck, Allan Porterfield, Nick Chaimov, Hartmut Kaiser, Allen D. Malony, Thomas Sterling, Rob Fowler, "An Autonomic Performance Environment for eXascale", *Journal of Supercomputing Frontiers and Innovations*, 2015.  http://superfri.org/superfri/article/view/64