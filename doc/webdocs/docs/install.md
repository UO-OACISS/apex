# Downloading APEX

APEX is open source, and available on Github at [http//github.com/khuck/xpress-apex](http//github.com/khuck/xpress-apex).

For stability, most users will want to download the most recent release of APEX (for example, v0.5):

```bash
wget https://github.com/khuck/xpress-apex/archive/v0.5.tar.gz
tar -xvzf v0.5.tar.gz
cd xpress-apex-0.5
```

Other users may want to work with the most recent code available, in which case you can clone the git repo:

```bash
git clone https://github.com/khuck/xpress-apex.git
cd xpress-apex
```

# Installation Option 1: Configuring and building APEX with bootstrap scripts

This option is useful for HPC resources where a configuration script already exists, such as Edison@NERSC, Cori@NERSC, Babbage@NERSC, etc.  To use this option, copy and modify ./bootstrap-configs/bootstrap-$arch.sh as necessary, and run it.

# Installation Option 2: Configuring and building APEX with CMake directly (recommended for most users)

APEX is built with CMake. The minimum CMake settings needed for APEX are:

* -DCMAKE_INSTALL_PREFIX= some path to an installation location
* -DCMAKE_BUILD_TYPE= one of Release, Debug, or RelWithDebInfo (recommended)

When building on Intel Phi, Boost is required if the compiler toolset does not include the latest GNU C++11 support.

* -DBOOST_ROOT= the path to a Boost installation, 1.54 or newer

**Note:** *If the BOOST_ROOT environment variable is set to a working Boost installation directory, CMake will find it automatically. If Boost is not installed locally, use the -DBUILD_BOOST=TRUE option, which will automatically download and build Boost as a subproject of APEX.*

The process for building APEX is:

1) Get the code (see above)

2) Enter the repo directory, make a build directory:

```bash
cd xpress-apex-0.5
mkdir build
cd build
```

3) configure using CMake:

```bash
cmake -DCMAKE_INSTALL_PREFIX=<installation-path> -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

If Boost is required (Intel Phi):

```bash
cmake -DBOOST_ROOT=<path-to-boost> -DCMAKE_INSTALL_PREFIX=<installation-path> -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

4) build with make:

```bash
make
make test
make doc
make install
```

# Other CMake settings, depending on your needs/wants

**Note 1:** *The **recommended** packages include:*

* **Active Harmony** - for autotuning policies
* **OMPT** - if OpenMP support is required ([See the OpenMP use case](usecases.md#openmp-example) for an example)
* **Binutils/BFD** - if your runtime/application uses instruction addresses to identify timers, e.g. HPX-5 and OpenMP
* **TAU** *or* **PAPI** - if you want post-mortem performance analysis ([See the TAU use case](usecases.md#with-tau) for an example) or your policies will require hardware counters ([See the PAPI use case](usecases.md#with-papi) for an example)
* **JEMalloc/TCMalloc** - if your application is not already using a heap manager - see Note 2, below

**Note 2:** *TCMalloc or JEMalloc will speed up memory allocations *significantly* in APEX (and in your application). HOWEVER, If your application already uses TCMalloc, JEMalloc or TBBMalloc, **DO NOT** configure APEX with TCMalloc or JEMalloc. They will be included at application link time, and may conflict with the version detected by and linked into APEX.*

There are several utility libraries that provide additional functionality in APEX. Not all libraries are required, but some are recommended.  For the following options, the default values are in *italics*.

* **-DBUILD\_BOOST=**
  TRUE or *FALSE*.  In the event that Boost isn't pre-installed on your system, this option forces CMake to download and build Boost as part of the APEX project.
* **-DUSE\_ACTIVEHARMONY=**
  *TRUE* or FALSE.  Active Harmony is a library that intelligently searches for parametric combinations to support adapting to heterogeneous and changing environments.  For more information, see <http://www.dyninst.org/harmony>.  APEX uses Active Harmony for runtime adaptation.
* **-DACTIVEHARMONY\_ROOT=**
  the path to Active Harmony, or set the ACTIVEHARMONY_ROOT environment variable before running cmake.  It should be noted that if Active Harmony is not specified and -DUSE_ACTIVEHARMONY is TRUE or not set, APEX will download and build Active Harmony as a CMake project. To disable Active Harmony entirely, specify -DUSE_ACTIVEHARMONY=FALSE.
* **-DBUILD\_ACTIVEHARMONY=**
  TRUE or *FALSE*.  Whether or not Active Harmony is installed on the system, this option forces CMake to automatically download and build Active Harmony as part of the APEX project.
* **-DUSE\_OMPT=**
  TRUE or *FALSE*.  OMPT is a proposed standard for OpenMP runtimes to provide callback hooks to performance tools. For more information, see <http://openmp.org/mp-documents/ompt-tr2.pdf>.  APEX has support for most OMPT OpenMP trace events. See [the OpenMP use case](usecases.md#openmp-example) for an example.
* **-DOMPT\_ROOT=**
  the path to OMPT, or set the OMPT_ROOT environment variable before running cmake.
* **-DBUILD\_OMPT=**
  TRUE or *FALSE*. Whether or not an OpenMP library with OMPT support is found by CMake, this option forces CMake to automatically download and build an OpenMP runtime with OMPT support as part of the APEX project.
* **-DUSE\_BFD=**
  TRUE or *FALSE*.  APEX uses libbfd to convert instruction addresses to source code locations. BFD support is useful for generating human-readable output for summaries and concurrency graphs. Libbfd is not required for runtime adaptation.
* **-DBFD\_ROOT=**
  path to Binutils, or set the BFD_ROOT environment variable.
* **-DBUILD\_BFD=**
  TRUE or FALSE.  Whether or not binutils is found by CMake, this option forces CMake to automatically download and build binutils as part of the APEX project.
* **-DUSE\_TAU=**
  TRUE or *FALSE*.  TAU (Tuning and Analysis Utilities) is a performance measurement and analysis framework for large scale parallel applications. For more information see <http://tau.uoregon.edu>.  APEX uses TAU to generate profiles for post-mortem performance analysis. See [the TAU use case](usecases.md#tau-example) for an example.
* **-DTAU\_ROOT=**
  path to TAU, or set the TAU_ROOT environment variable before running cmake.
* **-DTAU\_ARCH=**
  the TAU architecture, like x86_64, craycnl, mic_linux, bgq, etc.
* **-DTAU\_OPTIONS=**
  a TAU configuration with thread support, like -pthread or -icpc-pthread.
* **-DUSE\_RCR=**
  TRUE or *FALSE*.  RCR (Resource Centric Reflection) is a library for system monitoring of resources that require root access.  For more information, see <http://www.renci.org/wp-content/pub/techreports/TR-10-01.pdf>.  APEX uses RCR to access 'uncore' counters and system health information such as power and energy counters.
* **-DRCR\_ROOT=**
  the path to RCR, or set the RCR_ROOT environment variable.
* **-DUSE\_TCMALLOC=**
  TRUE or *FALSE*.  TCMalloc is a heap management library distributed as part of Google perftools. For more information, see <https://github.com/gperftools/gperftools>.  TCMalloc provides faster memory performance in multithreaded environments.
* **-DGPERFTOOLS\_ROOT=**
  path to gperftools (TCMalloc), or set the GPERFTOOLS_ROOT environment variable before running cmake.
* **-DUSE\_JEMALLOC=**
  TRUE or *FALSE*.  JEMalloc is a heap management library.  For more information, see <http://www.canonware.com/jemalloc/>.  JEMalloc provides faster memory performance in multithreaded environments.
* **-DJEMALLOC\_ROOT=**
  path to JEMalloc, or set the JEMALLOC_ROOT environment variable before running cmake.
* **-DUSE\_PAPI=**
  TRUE or *FALSE*.  PAPI (Performance Application Programming Interface) provides the tool designer and application engineer with a consistent interface and methodology for use of the performance counter hardware found in most major microprocessors.  For more information, see <http://icl.cs.utk.edu/papi/>.  APEX uses PAPI to optionally collect hardware counters for timed events.
* **-DPAPI\_ROOT=**
  some path to PAPI, or set the PAPI_ROOT environment variable before running cmake. See [the PAPI use case](usecases.md#papi-example) for an example.
* **-DUSE\_LM\_SENSORS=**
  TRUE or *FALSE*. Lm\_sensors (Linux Monitoring Sensors) is a library for monitoring hardware temperatures and fan speeds. For more information, see <https://en.wikipedia.org/wiki/Lm_sensors>.  APEX uses lm\_sensors to monitor hardware, where available.
* **-DBUILD\_EXAMPLES=**
  TRUE or *FALSE*. Whether or not to build the application examples in APEX.
* **-DBUILD\_TESTS=**
  *TRUE* or FALSE. Whether or not to build the APEX unit tests.
* **-DCMAKE\_C\_COMPILER=**
  *gcc*
* **-DCMAKE\_CXX\_COMPILER=**
  *g++*
* **-DCMAKE\_BUILD\_TYPE=**
  Release, *Debug*, RelWithDebInfo. Unfortunately, the cmake default (when not specified) is Debug. For faster performance, configure APEX to build RelWithDebInfo or Release.
* **-DBUILD\_SHARED\_LIBS=**
  TRUE or FALSE
* **-DUSE\_MPI=**
  TRUE or *FALSE*. Whether to build MPI global support and related examples.
* **-DMPI\_C\_INCLUDE\_PATH=**
  path to MPI headers
* **-DMPI\_CXX\_INCLUDE\_PATH=**
  path to MPI headers
* **-DMPI\_C\_LIBRARIES=**
  paths to MPI libraries, library names
* **-DMPI\_CXX\_LIBRARIES=**
  paths to MPI libraries, library names
* **-DMPI\_C\_COMPILER=**
  mpicc
* **-DMPI\_CXX\_COMPILER=**
  mpicxx
