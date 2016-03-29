#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

WORKDIR=/tmp

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

# None!

# OPTIONAL libraries - if left undefined, they likely won't be used.

#export BFD_ROOT=/usr
#export BFD_ROOT=/usr/at8.0
#export JEMALLOC_ROOT=/home3/khuck/install/jemalloc/3.6.0
#export GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#export RCR_ROOT=$HOME/src/RCRdaemon
export PAPI_ROOT=/usr/local/packages/papi/5.4.3/power8_gcc-4.9.2
#export TAU_ROOT=$HOME/src/tau2
export OMPT_ROOT=/usr/local/packages/LLVM-ompt/power8_gcc-4.9.2_debug
export ACTIVEHARMONY_ROOT=$HOME/install/activeharmony/4.5

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=TRUE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=${DIR}/../install/power8_gcc-4.9.2" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE 
cmake_other_settings="" # anything else?

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# run the main script
. $DIR/bootstrap-main.sh
