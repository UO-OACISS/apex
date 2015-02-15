#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

export BOOST_ROOT=$BOOST_DIR

# OPTIONAL libraries - if left undefined, they likely won't be used.

export BFD_ROOT=/usr
#GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#export RCR_ROOT=$HOME/src/RCRdaemon
#export PAPI_ROOT=/usr/local/papi/5.3.2
#export TAU_ROOT=$HOME/src/tau2-edison

# other CMake variables - for special situations / architectures / compilers.

# For edison: CMake will get these from the environment
export LDFLAGS="-dynamic -ldl" # cmake will pick this up
cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=FALSE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE
cmake_other_settings="\
-DCMAKE_C_COMPILER=cc \
-DCMAKE_CXX_COMPILER=CC \
" # anything else?

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
