#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

export BOOST_ROOT=$BOOST

# OPTIONAL libraries - if left undefined, they likely won't be used.

export BFD_ROOT=/home3/khuck/install/binutils-2.23.2
export JEMALLOC_ROOT=/home3/khuck/install/jemalloc/3.6.0
#export GPERFTOOLS_ROOT=/home3/khuck/install/gperftools/2.4
#export RCR_ROOT=/home3/khuck/src/RCRdaemon
#export PAPI_ROOT=/usr/local/papi/5.3.2
#export TAU_ROOT=/home3/khuck/src/tau2
export ACTIVEHARMONY_ROOT=$HOME/install/activeharmony/4.5
export OMPT_ROOT=$HOME/install/libiomp5

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=$DIR/../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=TRUE" # TRUE or FALSE
cmake_other_settings="" # anything else?

# runtime parameters for testing APEX with "make test"

export APEX_POLICY=1
export APEX_CONCURRENCY=0
export APEX_TAU=0

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# run the main script
. $DIR/bootstrap-main.sh
