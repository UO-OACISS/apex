#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

export CC=icc
export CXX=icpc
export FC=ifort
export MPI_C=mpiicc
export MPI_CXX=mpiicpc

# REQUIRED libraries

# BOOST_ROOT=$HOME/install/mic/boost/1.58.0
export BOOST_ROOT=$HOME/install/boost-1.58.0-intel

# OPTIONAL libraries - if left undefined, they likely won't be used.

export BFD_ROOT=$HOME/install/binutils-2.25-intel
export JEMALLOC_ROOT=$HOME/install/jemalloc/3.5.1
#GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#RCR_ROOT=$HOME/src/RCRdaemon
#PAPI_ROOT=/usr/local/papi/5.3.2
#TAU_ROOT=$HOME/install/tau-hpx
#OMPT_ROOT=$HOME/install/libiomp5
ACTIVEHARMONY_ROOT=$HOME/install/activeharmony/4.5

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=Debug" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE
cmake_other_settings="" # anything else?

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
