# Installing APEX

## Installation with HPX

APEX is integrated into the [HPX runtime](https://hpx.stellar-group.org), and is integrated into the HPX build system.  To enable APEX measurement with HPX, enable the following CMake flags:

```
-DHPX_WITH_APEX=TRUE
```

The `-DHPX_WITH_APEX_TAG=develop` can be used to indicate a specific release version of APEX, or to use a specific GitHub branch of APEX.  We recommend using the default configured version that comes with HPX (currently `v2.2.0`) or the `develop` branch.  Additional CMake flags include:

* `-DAPEX_WITH_LM_SENSORS=TRUE` to enable [LM sensors](https://hwmon.wiki.kernel.org/lm_sensors) support (assumed to be installed in default system paths)
* `-DAPEX_WITH_PAPI=TRUE` and `-DPAPI_ROOT=...` to enable [PAPI](https://icl.utk.edu/papi/) support
* `-DAPEX_WITH_BFD=TRUE` and `-DBFD_ROOT=...` *or* `-DBUILD_BFD=TRUE` to enable [Binutils](https://www.gnu.org/software/binutils/) support for converting function/lambda/instruction pointers to human-readable code regions.  For demangling of C++ symbols, `demangle.h` needs to be installed with the binutils headers (not typical in system installations). 
* `-DAPEX_WITH_MSR=TRUE` to enable [libmsr](https://github.com/LLNL/libmsr) support for RAPL power measurement (typically not needed, as RAPL support is natively handled where available)
* `-DAPEX_WITH_OTF2=TRUE` and `-DOTF2_ROOT=...` to enable [OTF2 tracing](https://www.vi-hps.org/projects/score-p/index.html) support
* `-DHPX_WITH_HPXMP=TRUE` to enable HPX OpenMP support and OMPT measurement support from APEX
* `-DAPEX_WITH_ACTIVEHARMONY=TRUE` and `-DACTIVEHARMONY_ROOT=...` to enable [Active Harmony](https://www.dyninst.org/harmony) support
* `-DAPEX_WITH_CUDA=TRUE` to enable CUPTI and/or NVML support.  This support requires a working `nvcc` compiler in your path.

## Standalone Installation

APEX is open source, and available on Github at <http://github.com/khuck/xpress-apex>.

For stability, most users will want to download the most recent release of APEX (for example, v2.2.0):

```bash
wget https://github.com/khuck/xpress-apex/archive/v2.2.0.tar.gz
tar -xvzf v2.2.0.tar.gz
cd xpress-apex-2.2.0
```

Other users may want to work with the most recent code available, in which case you can clone the git repo:

```bash
git clone https://github.com/khuck/xpress-apex.git
cd xpress-apex
```

### Configuring and building APEX with CMake

APEX is built with CMake. The minimum CMake settings needed for APEX are:

* `-DCMAKE_INSTALL_PREFIX=...` some path to an installation location
* `-DCMAKE_BUILD_TYPE=...` one of Release, Debug, or RelWithDebInfo (recommended)

Boost is **NOT** required to install APEX.  When building on Intel Phi, Boost is required if the compiler toolset does not include the latest GNU C++11 support.

* `-DBOOST_ROOT=...` the path to a Boost installation, 1.65 or newer

**Note:** *If the `BOOST_ROOT` environment variable is set to a working Boost installation directory, CMake will find it automatically.*

The process for building APEX is:

1) Get the code (see above)

2) Enter the repo directory, make a build directory:

```bash
cd xpress-apex-2.2.0
mkdir build
cd build
```

3) configure using CMake:

```bash
cmake -DCMAKE_INSTALL_PREFIX=<installation-path> -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

If Boost is required (Intel Phi **only**):

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

### Other CMake settings, depending on your needs/wants

**Note 1:** *The **recommended** packages include:*

* **Active Harmony** - for autotuning policies
* **OMPT** - if OpenMP support is required ([See the OpenMP use case](usecases.md#openmp-example) for an example)
* **Binutils/BFD** - if your runtime/application uses instruction addresses to identify timers, e.g. OpenMP
* **PAPI** - if you want hardware counter support ([See the PAPI use case](usecases.md#with-papi) for an example)
* **JEMalloc/TCMalloc** - if your application is not already using a heap manager - see Note 2, below
* **CUDA** - if your application uses CUDA, APEX will use CUPTI/NVML to measure GPU activity

**Note 2:** *TCMalloc or JEMalloc will potentially speed up memory allocations **significantly** in APEX (and in your application). HOWEVER, If your application already uses TCMalloc, JEMalloc or TBBMalloc, **DO NOT** configure APEX with TCMalloc or JEMalloc. They will be included at application link time, and may conflict with the version detected by and linked into APEX.*

There are several utility libraries that provide additional functionality in APEX. Not all libraries are required, but some are recommended.  For the following options, the default values are in *italics*.

* `-DUSE_ACTIVEHARMONY=`
  *TRUE* or FALSE.  Active Harmony is a library that intelligently searches for parametric combinations to support adapting to heterogeneous and changing environments.  For more information, see <http://www.dyninst.org/harmony>.  APEX uses Active Harmony for runtime adaptation.
* `-DACTIVEHARMONY_ROOT=`
  the path to Active Harmony, or set the ACTIVEHARMONY_ROOT environment variable before running cmake.  It should be noted that if Active Harmony is not specified and -DUSE_ACTIVEHARMONY is TRUE or not set, APEX will download and build Active Harmony as a CMake project. To disable Active Harmony entirely, specify -DUSE_ACTIVEHARMONY=FALSE.
* `-DBUILD_ACTIVEHARMONY=`
  TRUE or *FALSE*.  Whether or not Active Harmony is installed on the system, this option forces CMake to automatically download and build Active Harmony as part of the APEX project.
* `-DUSE_OMPT=`
  TRUE or *FALSE*.  OMPT is a proposed standard for OpenMP runtimes to provide callback hooks to performance tools. For more information, see the [OpenMP specification](https://www.openmp.org/specifications/) v5.0 or newer.  APEX has support for most OMPT OpenMP trace events. See [the OpenMP use case](usecases.md#openmp-example) for an example.  Some compilers (Clang 10+, Intel 19+, IBM XL 16+) include OMPT support already, and APEX will use the built-in support.  For GCC, older Clang and older Intel Compilers, APEX can build and use the [LLVM OpenMP runtime](https://github.com/llvm-mirror/openmp) which provides KMP and GOMP API calls for those compilers.
* `-DOMPT_ROOT=`
  the path to OMPT, or set the OMPT_ROOT environment variable before running cmake.
* `-DBUILD_OMPT=`
  TRUE or *FALSE*. Whether or not an OpenMP library with OMPT support is found by CMake, this option forces CMake to automatically download and build an OpenMP runtime with OMPT support as part of the APEX project.  In most cases, only relevant for gcc/g++.
* `-DUSE_BFD=`
  TRUE or *FALSE*.  APEX uses libbfd (Binutils) to convert instruction addresses to source code locations. BFD support is useful for generating human-readable output for summaries and concurrency graphs. Libbfd is not required for runtime adaptation.  For more information, see <https://www.gnu.org/software/binutils/>.
* `-DBFD_ROOT=`
  path to Binutils, or set the BFD_ROOT environment variable.
* `-DBUILD_BFD=`
  TRUE or FALSE.  Whether or not binutils is found by CMake, this option forces CMake to automatically download and build binutils as part of the APEX project.
* `-DUSE_TCMALLOC=`
  TRUE or *FALSE*.  TCMalloc is a heap management library distributed as part of Google perftools. For more information, see <https://github.com/gperftools/gperftools>.  TCMalloc provides faster memory performance in multithreaded environments.
* `-DGPERFTOOLS_ROOT=`
  path to gperftools (TCMalloc), or set the GPERFTOOLS_ROOT environment variable before running cmake.
* `-DUSE_JEMALLOC=`
  TRUE or *FALSE*.  JEMalloc is a heap management library.  For more information, see <http://www.canonware.com/jemalloc/>.  JEMalloc provides faster memory performance in multithreaded environments.
* `-DJEMALLOC\_ROOT=`
  path to JEMalloc, or set the JEMALLOC_ROOT environment variable before running cmake.
* `-DUSE_PAPI=`
  TRUE or *FALSE*.  PAPI (Performance Application Programming Interface) provides the tool designer and application engineer with a consistent interface and methodology for use of the performance counter hardware found in most major microprocessors.  For more information, see <http://icl.cs.utk.edu/papi/>.  APEX uses PAPI to optionally collect hardware counters for timed events.
* `-DPAPI_ROOT=`
  some path to PAPI, or set the PAPI_ROOT environment variable before running cmake. See [the PAPI use case](usecases.md#papi-example) for an example.
* `-DUSE_LM_SENSORS=`
  TRUE or *FALSE*. Lm\_sensors (Linux Monitoring Sensors) is a library for monitoring hardware temperatures and fan speeds. For more information, see <https://en.wikipedia.org/wiki/Lm_sensors>.  APEX uses lm\_sensors to monitor hardware, where available.
* `-DBUILD_EXAMPLES=`
  TRUE or *FALSE*. Whether or not to build the application examples in APEX.
* `-DBUILD_TESTS=`
  *TRUE* or FALSE. Whether or not to build the APEX unit tests.
* `-DUSE_MPI=`
  TRUE or *FALSE*. Whether to build MPI global support and related examples.

### Other CMake variables of interest

* `-DCMAKE_C_COMPILER=`
  *gcc* or the `CC` environment variable setting
* `-DCMAKE_CXX_COMPILER=`
  *g++* or the `CXX` environment variable setting
* `-DCMAKE_BUILD`_TYPE=`
  Release, *Debug*, RelWithDebInfo. Unfortunately, the cmake default (when not specified) is Debug. For faster performance, configure APEX to build RelWithDebInfo or Release.
* `-DBUILD_SHARED_LIBS=`
  TRUE or FALSE
