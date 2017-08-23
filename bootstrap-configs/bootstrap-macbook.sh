#!/bin/bash -e

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# OPTIONAL libraries - if left undefined, they likely won't be used.

#BFD_ROOT=$HOME/install/binutils-2.23.2
#BFD_ROOT=/opt/local
#GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#RCR_ROOT=$HOME/src/RCRdaemon
PAPI_ROOT=/usr/local/papi/5.4.3
# export TAU_ROOT=$HOME/src/tau2
# export TAU_ARCH=apple
# export TAU_OPTIONS=-pthread
unset TAU_ROOT

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=TRUE -DBUILD_EXAMPLES=TRUE" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
#cmake_use_codeblocks="-G Xcode" # if you want to debug in Xcode
#cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
#cmake_use_codeblocks="-G \"Eclipse CDT4 - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE
cmake_other_settings="-DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx -DOTF2_ROOT=$HOME/install/otf2/2.0" # anything else?

# runtime parameters for testing APEX with "make test"

export APEX_POLICY=1
export APEX_CONCURRENCY=0
export APEX_TAU=0

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# run the main script
. $DIR/bootstrap-main.sh
