![Lame APEX logo](doc/logo-cropped.png)

Please Note:
===========
*All documentation is outdated and currently going through a review and update.  Thanks for your patience.*

Build Status:
===========
The [Buildbot](http://omega.nic.uoregon.edu:8010/#/grid) continuous integration
service tracks the current build status on several platforms and compilers:

[![default-release](http://omega.nic.uoregon.edu:8010/badges/A-default-release.svg?left_text=default-release)](http://omega.nic.uoregon.edu:8010/#/)
[![base-release](http://omega.nic.uoregon.edu:8010/badges/B-base-release.svg?left_text=base-release)](http://omega.nic.uoregon.edu:8010/#/)
[![malloc-release](http://omega.nic.uoregon.edu:8010/badges/C-malloc-release.svg?left_text=malloc-release)](http://omega.nic.uoregon.edu:8010/#/)
[![bfd-release](http://omega.nic.uoregon.edu:8010/badges/D-bfd-release.svg?left_text=bfd-release)](http://omega.nic.uoregon.edu:8010/#/)
[![ah-release](http://omega.nic.uoregon.edu:8010/badges/E-ah-release.svg?left_text=ah-release)](http://omega.nic.uoregon.edu:8010/#/)
[![ompt-release](http://omega.nic.uoregon.edu:8010/badges/F-ompt-release.svg?left_text=ompt-release)](http://omega.nic.uoregon.edu:8010/#/)
[![papi-release](http://omega.nic.uoregon.edu:8010/badges/G-papi-release.svg?left_text=papi-release)](http://omega.nic.uoregon.edu:8010/#/)
[![mpi-release](http://omega.nic.uoregon.edu:8010/badges/H-mpi-release.svg?left_text=mpi-release)](http://omega.nic.uoregon.edu:8010/#/)
[![otf-release](http://omega.nic.uoregon.edu:8010/badges/I-otf-release.svg?left_text=otf-release)](http://omega.nic.uoregon.edu:8010/#/)
[![install-release](http://omega.nic.uoregon.edu:8010/badges/J-install-release.svg?left_text=install-release)](http://omega.nic.uoregon.edu:8010/#/)
[![hpx-release](http://omega.nic.uoregon.edu:8010/badges/K-hpx-release.svg?left_text=hpx-release)](http://omega.nic.uoregon.edu:8010/#/)

[![default-debug](http://omega.nic.uoregon.edu:8010/badges/A-default-debug.svg?left_text=default-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![base-debug](http://omega.nic.uoregon.edu:8010/badges/B-base-debug.svg?left_text=base-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![malloc-debug](http://omega.nic.uoregon.edu:8010/badges/C-malloc-debug.svg?left_text=malloc-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![bfd-debug](http://omega.nic.uoregon.edu:8010/badges/D-bfd-debug.svg?left_text=bfd-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![ah-debug](http://omega.nic.uoregon.edu:8010/badges/E-ah-debug.svg?left_text=ah-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![ompt-debug](http://omega.nic.uoregon.edu:8010/badges/F-ompt-debug.svg?left_text=ompt-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![papi-debug](http://omega.nic.uoregon.edu:8010/badges/G-papi-debug.svg?left_text=papi-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![mpi-debug](http://omega.nic.uoregon.edu:8010/badges/H-mpi-debug.svg?left_text=mpi-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![otf-debug](http://omega.nic.uoregon.edu:8010/badges/I-otf-debug.svg?left_text=otf-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![install-debug](http://omega.nic.uoregon.edu:8010/badges/J-install-debug.svg?left_text=install-debug)](http://omega.nic.uoregon.edu:8010/#/)
[![hpx-debug](http://omega.nic.uoregon.edu:8010/badges/K-hpx-debug.svg?left_text=hpx-debug)](http://omega.nic.uoregon.edu:8010/#/)

Overview:
=========

One of the key components of the XPRESS project is a new approach to performance observation, measurement, analysis and runtime decision making in order to optimize performance. The particular challenges of accurately measuring the performance characteristics of ParalleX [\[1\]](#footnote1) applications (as well as other asynchronous multitasking runtime architectures) requires a new approach to parallel performance observation. The standard model of multiple operating system processes and threads observing themselves in a first-person manner while writing out performance profiles or traces for offline analysis will not adequately capture the full execution context, nor provide opportunities for runtime adaptation within OpenX. The approach taken in the XPRESS project is a new performance measurement system, called (Autonomic Performance Environment for eXascale). APEX includes methods for information sharing between the layers of the software stack, from the hardware through operating and runtime systems, all the way to domain specific or legacy applications. The performance measurement components incorporate relevant information across stack layers, with merging of third-person performance observation of node-level and global resources, remote processes, and both operating and runtime system threads.  For a complete academic description of APEX, see the publication "APEX: An Autonomic Performance Environment for eXascale"  [\[2\]](#footnote2).

In short, APEX is an introspection and runtime adaptation library for asynchronous multitasking runtime systems. However, APEX is not *only* useful for AMT/AMR runtimes - it can be used by any application wanting to perform runtime adaptation to deal with heterogeneous and/or variable environments.

Introspection
-------------
APEX provides an API for measuring actions within a runtime. The API includes methods for timer start/stop, as well as sampled counter values. APEX is designed to be integrated into a runtime, library and/or application and provide performance introspection for the purpose of runtime adaptation. While APEX *can* provide rudimentary post-mortem performance analysis measurement, there are many other performance measurement tools that perform that task much better (such as [TAU](http://tau.uoregon.edu)).  That said, APEX includes an event listener that integrates with the TAU measurement system, so APEX events can be forwarded to TAU and collected in a TAU profile and/or trace to be used for post-mortem performance anlaysis.

Runtime Adaptation
------------------
APEX provides a mechanism for dynamic runtime behavior, either for autotuning or adaptation to changing environment.  The infrastruture that provides the adaptation is the Policy Engine, which executes policies either periodically or triggered by events. The policies have access to the performance state as observed by the APEX introspection API. APEX is integrated with [Active Harmony](http://www.dyninst.org/harmony) to provide dynamic search for autotuning.

Documentation
=============

Full user documentation is available here: http://khuck.github.io/xpress-apex.

The source code is instrumented with Doxygen comments, and the API reference manual can be generated by executing `make doc` in the build directory, after CMake configuration.  [A fairly recent version of the API reference documentation is also available here] (http://www.nic.uoregon.edu/~khuck/apex_docs/doc/html/index.html).

Installation
============

[Full installation documentation is available here] (http://khuck.github.io/xpress-apex). Below is a quickstart for the impatient...

Please Note:
------------
*These instructions are for building the stand-alone APEX library.  For instructions on building APEX with HPX, please see [http://khuck.github.io/xpress-apex/usage](http://khuck.github.io/xpress-apex/usage)*


To build APEX stand-alone (to use with OpenMP, OpenACC, CUDA, Kokkos, TBB, C++ threads, etc.) do the following:

```
git clone https://github.com/khuck/xpress-apex.git
cd xpress-apex
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=TRUE ..
make -j
```

Runtime Usage
-------------

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

Supported Runtime Systems
=========================

HPX (Louisiana State University)
---------------------------------

HPX (High Performance ParalleX) is the original implementation of the ParalleX model. Developed and maintained by the Ste||ar Group at Louisiana State University, HPX is implemented in C++. For more information, see [http://stellar.cct.lsu.edu/tag/hpx/](http://stellar.cct.lsu.edu/tag/hpx/).  For a tutorial on HPX with APEX (presented at SC'17, Austin TX) see [http://www.nic.uoregon.edu/~khuck/SC17-HPX-APEX.pdf](http://www.nic.uoregon.edu/~khuck/SC17-HPX-APEX.pdf).  The integration specification is available [here](http://www.nic.uoregon.edu/~khuck/Phylanx/2019_report.pdf).

HPX5 (Indiana University)
-------------------------

HPX-5 (High Performance ParalleX) is a second implementation of the ParalleX model. Developed and maintained by the CREST Group at Indiana University, HPX-5 is implemented in C.  For more information, see [https://hpx.crest.iu.edu](https://hpx.crest.iu.edu).

OpenMP
------

The OpenMP API supports multi-platform shared-memory parallel programming in C/C++ and Fortran. The OpenMP API defines a portable, scalable model with a simple and flexible interface for developing parallel applications on platforms from the desktop to the supercomputer.  For more information, see [http://openmp.org/](http://openmp.org/).

OpenACC
-------

OpenACC is a user-driven directive-based performance-portable parallel programming model. It is designed for scientists and engineers interested in porting their codes to a wide-variety of heterogeneous HPC hardware platforms and architectures with significantly less programming effort than required with a low-level model. The OpenACC specification supports C, C++, Fortran programming languages and multiple hardware architectures including X86 & POWER CPUs, and NVIDIA GPUs.

Kokkos
------

Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management. Kokkos is designed to target complex node architectures with N-level memory hierarchies and multiple types of execution resources. It currently can use CUDA, HPX, OpenMP and Pthreads as backend programming models with several other backends in development.

CUDA
----

CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.

References
==========
<a name="footnote1">[1]</a> Thomas Sterling, Daniel Kogler, Matthew Anderson, and Maciej Brodowicz. "SLOWER: A performance model for Exascale computing". *Supercomputing Frontiers and Innovations*, 1:42–57, September 2014.  http://superfri.org/superfri/article/view/10

<a name="footnote2">[2]</a> Kevin A. Huck, Allan Porterfield, Nick Chaimov, Hartmut Kaiser, Allen D. Malony, Thomas Sterling, Rob Fowler. "An Autonomic Performance Environment for eXascale", *Journal of Supercomputing Frontiers and Innovations*, 2015.  http://superfri.org/superfri/article/view/64
