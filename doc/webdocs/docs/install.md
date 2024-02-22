# Installing APEX

## Installation with HPX

APEX is integrated into the [HPX runtime](https://hpx.stellar-group.org), and is integrated into the HPX build system.  To enable APEX measurement with HPX, enable the following CMake flags:

```
-DHPX_WITH_APEX=TRUE
```

The `-DHPX_WITH_APEX_TAG=develop` can be used to indicate a specific release version of APEX, or to use a specific GitHub branch of APEX.  We recommend using the default configured version that comes with HPX (currently `v2.6.5`) or the `develop` branch.  Additional CMake flags include:

* `-DAPEX_WITH_LM_SENSORS=TRUE` to enable [LM sensors](https://hwmon.wiki.kernel.org/lm_sensors) support (assumed to be installed in default system paths)
* `-DAPEX_WITH_PAPI=TRUE` and `-DPAPI_ROOT=...` to enable [PAPI](https://icl.utk.edu/papi/) support
* `-DAPEX_WITH_BFD=TRUE` and `-DBFD_ROOT=...` *or* `-DAPEX_BUILD_BFD=TRUE` to enable [Binutils](https://www.gnu.org/software/binutils/) support for converting function/lambda/instruction pointers to human-readable code regions.  For demangling of C++ symbols, `demangle.h` needs to be installed with the binutils headers (not typical in system installations).
* `-DAPEX_WITH_MSR=TRUE` to enable [libmsr](https://github.com/LLNL/libmsr) support for RAPL power measurement (typically not needed, as RAPL support is natively handled where available)
* `-DAPEX_WITH_OTF2=TRUE` and `-DOTF2_ROOT=...` to enable [OTF2 tracing](https://www.vi-hps.org/projects/score-p/index.html) support
* `-DHPX_WITH_HPXMP=TRUE` to enable HPX OpenMP support and OMPT measurement support from APEX
* `-DAPEX_WITH_ACTIVEHARMONY=TRUE` and `-DACTIVEHARMONY_ROOT=...` to enable [Active Harmony](https://www.dyninst.org/harmony) support
* `-DAPEX_WITH_CUDA=TRUE` to enable CUPTI and/or NVML support.  Examples require a working `nvcc` compiler in your path.

## Standalone Installation

APEX is open source, and available on Github at <http://github.com/UO-OACISS/apex>.

For stability, most users will want to download [the most recent release](https://github.com/UO-OACISS/apex/releases) of APEX (for example, v2.6.5):

```bash
wget https://github.com/UO-OACISS/apex/archive/refs/tags/v2.6.5.tar.gz
tar -xvzf v2.6.5.tar.gz
cd apex-2.6.5
```

Other users may want to work with the most recent code available, in which case you can clone the git repo:

```bash
git clone https://github.com/UO-OACISS/apex.git
cd apex
```

### Configuring and building APEX with Spack

APEX can be installed with the [Spack package management tool](https://spack.readthedocs.io/en/latest/). See `spack info apex` for details. You should see something like this:

```
CMakePackage:   apex

Description:
    Autonomic Performance Environment for eXascale (APEX).

Homepage: https://uo-oaciss.github.io/apex

Preferred version:
    2.6.3      https://github.com/UO-OACISS/apex/archive/v2.6.3.tar.gz

Safe versions:
    develop    [git] https://github.com/UO-OACISS/apex on branch develop
    master     [git] https://github.com/UO-OACISS/apex on branch master
    2.6.3      https://github.com/UO-OACISS/apex/archive/v2.6.3.tar.gz
    2.6.2      https://github.com/UO-OACISS/apex/archive/v2.6.2.tar.gz
    2.6.1      https://github.com/UO-OACISS/apex/archive/v2.6.1.tar.gz
    2.6.0      https://github.com/UO-OACISS/apex/archive/v2.6.0.tar.gz

Deprecated versions:
    2.5.1      https://github.com/UO-OACISS/apex/archive/v2.5.1.tar.gz
    2.5.0      https://github.com/UO-OACISS/apex/archive/v2.5.0.tar.gz
    2.4.1      https://github.com/UO-OACISS/apex/archive/v2.4.1.tar.gz
    2.4.0      https://github.com/UO-OACISS/apex/archive/v2.4.0.tar.gz
    2.3.2      https://github.com/UO-OACISS/apex/archive/v2.3.2.tar.gz
    2.3.1      https://github.com/UO-OACISS/apex/archive/v2.3.1.tar.gz
    2.3.0      https://github.com/UO-OACISS/apex/archive/v2.3.0.tar.gz
    2.2.0      https://github.com/UO-OACISS/apex/archive/v2.2.0.tar.gz

Variants:
    activeharmony [true]        false, true
        Enables Active Harmony support
    binutils [false]            false, true
        Enables Binutils support
    boost [false]               false, true
        Enables Boost support
    build_system [cmake]        cmake
        Build systems supported by the package
    cuda [false]                false, true
        Enables CUDA support
    examples [false]            false, true
        Build Examples
    gperftools [false]          false, true
        Enables Google PerfTools TCMalloc support
    hip [false]                 false, true
        Enables ROCm/HIP support
    jemalloc [false]            false, true
        Enables JEMalloc support
    lmsensors [false]           false, true
        Enables LM-Sensors support
    mpi [false]                 false, true
        Enables MPI support
    openmp [false]              false, true
        Enables OpenMP support
    otf2 [true]                 false, true
        Enables OTF2 support
    papi [false]                false, true
        Enables PAPI support
    plugins [true]              false, true
        Enables Policy Plugin support
    sycl [false]                false, true
        Enables Intel SYCL support (Level0)
    tests [false]               false, true
        Build Unit Tests

    when build_system=cmake
      build_type [Release]      Debug, MinSizeRel, RelWithDebInfo, Release
          CMake build type
      generator [make]          none
          the build system generator to use

    when build_system=cmake ^cmake@3.9:
      ipo [false]               false, true
          CMake interprocedural optimization

Build Dependencies:
    activeharmony  boost  cuda     gmake       hip       lm-sensors  ninja  papi          roctracer-dev  zlib-api
    binutils       cmake  gettext  gperftools  jemalloc  mpi         otf2   rocm-smi-lib  sycl

Link Dependencies:
    activeharmony  binutils  boost  cuda  gettext  gperftools  hip  jemalloc  lm-sensors  mpi  otf2  papi  rocm-smi-lib  roctracer-dev  sycl  zlib-api

Run Dependencies:
    None

Licenses:
    None
```

### Configuring and building APEX with CMake

APEX is built with CMake. The minimum CMake settings needed for APEX are:

* `-DCMAKE_INSTALL_PREFIX=...` some path to an installation location
* `-DCMAKE_BUILD_TYPE=...` one of Release, Debug, or RelWithDebInfo (Release recommended)

<!--
Boost is **NOT** required to install APEX.  When building on Intel Phi, Boost is required if the compiler toolset does not include the latest GNU C++11 support. (may not still be true)

* `-DBOOST_ROOT=...` the path to a Boost installation, 1.65 or newer

**Note:** *If the `BOOST_ROOT` environment variable is set to a working Boost installation directory, CMake will find it automatically.*
-->

The process for building APEX is:

1) Get the code (see above)

2) Enter the repo directory:

```bash
cd apex-2.6.5
```

3) configure using CMake:

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=<installation-path> -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

4) build with cmake:

```bash
cmake --build build
# Run tests, if desired
ctest --test-dir build
# Build documentation, if desired
cd build ; make doc ; cd ..
# Install, if desired
cmake --install install
```

### Other CMake settings, depending on your needs/wants

**Note 1:** *The **recommended** packages include:*

* **Active Harmony** - for autotuning policies (optional, no longer recommended)
* **OMPT** - if OpenMP support is required ([See the OpenMP use case](usecases.md#openmp-example) for an example) and your compiler supports OpenMP-Tools. *note: GCC does not support OpenMP-Tools, and has no plans to as of January 2024.* Compilers _known_ to support OMPT include Clang/LLVM, Intel, NVIDIA, AMD Clang.
* **Binutils/BFD** - if your runtime/application uses instruction addresses to identify timers, e.g. OpenMP, CUDA, HIP, OneAPI, OpenACC, etc.
* **PAPI** - if you want hardware counter support ([See the PAPI use case](usecases.md#with-papi) for an example)
* **JEMalloc/TCMalloc** - if your application is not already using a heap manager - see Note 2, below
* **CUDA** - if your application uses CUDA, APEX will use CUPTI/NVML to measure GPU activity
* **ROCM** - if your application uses HIP/ROCm, APEX will use Rocprofiler/Roctracer/ROC-SMI to measure GPU activity
* **OneAPI** - if your application uses Intel SYCL, APEX will use OneAPI/LevelZero to measure GPU activity

**Note 2:** *TCMalloc or JEMalloc will potentially speed up memory allocations **significantly** in APEX (and in your application). HOWEVER, If your application already uses TCMalloc, JEMalloc or TBBMalloc, **DO NOT** configure APEX with TCMalloc or JEMalloc. They will be included at application link time, and may conflict with the version detected by and linked into APEX. If you got some kind of tcmalloc crash/error at startup, please preload the dependent tcmalloc shared object library with '--apex:preload /path/to/libtcmalloc.so'.*

There are several utility libraries that provide additional functionality in APEX. Not all libraries are required, but some are recommended.  For the following options, the default values are in *italics*.

<!-- generic stuff -->
* `-DAPEX_BUILD_EXAMPLES=` TRUE or *FALSE*. Whether or not to build the application examples in APEX.
* `-DAPEX_BUILD_TESTS=` TRUE or *FALSE*. Whether or not to build the APEX unit tests.
<!-- active harmony -->
* `-DAPEX_WITH_ACTIVEHARMONY=` TRUE or *FALSE*.  Active Harmony is a library that intelligently searches for parametric combinations to support adapting to heterogeneous and changing environments.  For more information, see <http://www.dyninst.org/harmony>.  APEX uses Active Harmony for runtime adaptation.
* `-DACTIVEHARMONY_ROOT=` the path to Active Harmony, or set the ACTIVEHARMONY_ROOT environment variable before running cmake.  It should be noted that if Active Harmony is not specified and -DAPEX_WITH_ACTIVEHARMONY is TRUE or not set, APEX will download and build Active Harmony as a CMake project. To disable Active Harmony entirely, specify -DAPEX_WITH_ACTIVEHARMONY=FALSE.
* `-DAPEX_BUILD_ACTIVEHARMONY=` TRUE or *FALSE*.  Whether or not Active Harmony is installed on the system, this option forces CMake to automatically download and build Active Harmony as part of the APEX project.
<!-- binutils -->
* `-DAPEX_WITH_BFD=` TRUE or *FALSE*.  APEX uses libbfd (Binutils) to convert instruction addresses to source code locations. BFD support is useful for generating human-readable output for summaries and concurrency graphs. Libbfd is not required for runtime adaptation.  For more information, see <https://www.gnu.org/software/binutils/>.
* `-DBFD_ROOT=` path to Binutils, or set the BFD_ROOT environment variable.
* `-DAPEX_BUILD_BFD=` TRUE or *FALSE*.  Whether or not binutils is found by CMake, this option forces CMake to automatically download and build binutils as part of the APEX project.
<!-- cuda -->
* `-DAPEX_WITH_CUDA=` TRUE or *FALSE*. APEX uses CUPTI to measure CUDA kernels and API calls, and/or NVML support to monitor the GPU activity passively.
* `-DCUDAToolkit_ROOT=` the path to the CUDA installation, if necessary.
<!-- hip -->
* `-DAPEX_WITH_HIP=` TRUE or *FALSE*. APEX uses Rocprofiler and Roctracer to measure HIP kernels and API calls, and/or ROCM-SMI support to monitor the GPU activity passively.
* `-DROCM_ROOT=` the path to the ROCm installation, if necessary.
<!-- kokkos -->
* `-DAPEX_WITH_KOKKOS=` *TRUE* or FALSE.
* `-DKokkos_ROOT=` the path to the Kokkos installation, if necessary. APEX will grab Kokkos as a submodule if not found, only the headers are needed.
<!-- jemalloc -->
* `-DAPEX_WITH_JEMALLOC=` TRUE or *FALSE*.  JEMalloc is a heap management library.  For more information, see <http://www.canonware.com/jemalloc/>.  JEMalloc provides faster memory performance in multithreaded environments.
* `-DJEMALLOC\_ROOT=` path to JEMalloc, or set the JEMALLOC_ROOT environment variable before running cmake.
<!-- level0 -->
* `-DAPEX_WITH_LEVEL0=` TRUE or *FALSE*. APEX uses Level0 to measure Intel SYCL kernels and API calls and to monitor the GPU activity passively.
<!-- lm sensors -->
* `-DAPEX_WITH_LM_SENSORS=` TRUE or *FALSE*. Lm\_sensors (Linux Monitoring Sensors) is a library for monitoring hardware temperatures and fan speeds. For more information, see <https://en.wikipedia.org/wiki/Lm_sensors>.  APEX uses lm\_sensors to monitor hardware, where available.
<!-- mpi -->
* `-DAPEX_WITH_MPI=` TRUE or *FALSE*. Whether to build MPI global support and related examples.
<!-- openmp -->
* `-DAPEX_WITH_OMPT=` TRUE or *FALSE*.  OMP-Tools is the 5.0+ standard for OpenMP runtimes to provide callback hooks to performance tools. For more information, see the [OpenMP specification](https://www.openmp.org/specifications/) v5.0 or newer.  APEX has support for most OMPT OpenMP trace events. See [the OpenMP use case](usecases.md#openmp-example) for an example.  Some compilers (Clang 10+, Intel 19+, IBM XL 16+) include OMPT support already, and APEX will use the built-in support.  <!--For GCC, older Clang and older Intel Compilers, APEX can build and use the [LLVM OpenMP runtime](https://github.com/llvm-mirror/openmp) which provides KMP and GOMP API calls for those compilers.-->
<!-- * `-DOMPT_ROOT=` the path to OMPT, or set the OMPT_ROOT environment variable before running cmake. -->
<!-- otf2 -->
* `-DAPEX_WITH_OTF2=` TRUE or *FALSE*. Used to enable [OTF2 tracing](https://www.vi-hps.org/projects/score-p/index.html) support for the Vampir trace visualization tool.
* `-DOTF2_ROOT=` path to an OTF2 installation.
* `-DAPEX_BUILD_OTF2=` TRUE or *FALSE*.  If OTF2 is not found by CMake, this option forces CMake to automatically download and build binutils as part of the APEX project.
<!-- papi -->
* `-DAPEX_WITH_PAPI=` TRUE or *FALSE*.  PAPI (Performance Application Programming Interface) provides the tool designer and application engineer with a consistent interface and methodology for use of the performance counter hardware found in most major microprocessors.  For more information, see <http://icl.cs.utk.edu/papi/>.  APEX uses PAPI to optionally collect hardware counters for timed events.
* `-DPAPI_ROOT=` some path to PAPI, or set the PAPI_ROOT environment variable before running cmake. See [the PAPI use case](usecases.md#papi-example) for an example.
<!-- perfetto -->
* `-DAPEX_WITH_PERFETTO=` TRUE or *FALSE*. Enables native Perfetto trace support, increases build/link time significantly. Only used if you want native Perfetto output support, otherwise APEX will write compressed JSON output of the same data (which is actually smaller than the binary native format).
<!-- phiprof -->
<!-- plugins -->
* `-DAPEX_WITH_PLUGINS=` *TRUE* or FALSE. Enables APEX policy plugin support.
<!-- starpu -->
<!-- tcmalloc -->
* `-DAPEX_WITH_TCMALLOC=` TRUE or *FALSE*.  TCMalloc is a heap management library distributed as part of Google perftools. For more information, see <https://github.com/gperftools/gperftools>.  TCMalloc provides faster memory performance in multithreaded environments.
* `-DGPERFTOOLS_ROOT=` path to gperftools (TCMalloc), or set the `GPERFTOOLS_ROOT` environment variable before running cmake.

  <!-- Deprecated
* `-DAPEX_BUILD_OMPT=` TRUE or *FALSE*. Whether or not an OpenMP library with OMPT support is found by CMake, this option forces CMake to automatically download and build an OpenMP runtime with OMPT support as part of the APEX project.  In most cases, only relevant for gcc/g++.
  -->

### Other CMake variables of interest

For any others not listed, see <https://github.com/UO-OACISS/apex/blob/develop/cmake/Modules/APEX_DefaultOptions.cmake>
