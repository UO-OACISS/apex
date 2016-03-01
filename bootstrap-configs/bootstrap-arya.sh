#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

BOOST_ROOT=$HOME/hpx/boost_1_56_0

# OPTIONAL libraries - if left undefined, they likely won't be used.

BFD_ROOT=$HOME/src/tau2/x86_64/binutils-2.23.2
#GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#RCR_ROOT=$HOME/src/RCRdaemon
#PAPI_ROOT=/usr/local/papi/5.3.2
#TAU_ROOT=$HOME/install/tau-hpx
OMPT_ROOT=/home/users/nchaimov/LLVM-openmp/build-gcc
ACTIVEHARMONY_ROOT=$HOME/hpx/activeharmony

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=TRUE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE
cmake_other_settings="-DCMAKE_C_COMPILER=gcc-4.9 -DCMAKE_CXX_COMPILER=g++-4.9 -DUSE_PLUGINS=TRUE -DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DUSE_ACTIVEHARMONY=TRUE -DACTIVEHARMONY_ROOT=$ACTIVEHARMONY_ROOT -DUSE_OMPT=TRUE -DUSE_JEMALLOC=TRUE -DUSE_BFD=1 -DUSE_MSR=1 -DMSR_ROOT=$HOME/src/libmsr/inst"  # anything else?
#cmake_other_settings="-DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DUSE_ACTIVEHARMONY=TRUE -DUSE_OMPT=TRUE" # anything else?

# runtime parameters for testing APEX with "make test"

export APEX_POLICY=1
export APEX_THROTTLING=1
export APEX_CONCURRENCY=0
export APEX_TAU=0

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# run the main script
. $DIR/bootstrap-main.sh
