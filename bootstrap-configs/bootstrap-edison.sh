#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

export BOOST_ROOT=$BOOST_DIR

# OPTIONAL libraries - if left undefined, they likely won't be used.

# export BFD_ROOT=/usr # CMake will find it automatically.
#GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4 # not necessary, because HPX uses JEMalloc
#export JEMALLOC_ROOT=$HOME/install/jemalloc/3.6.0 # not strictly necessary, if runtime uses JEMalloc
#export RCR_ROOT=$HOME/src/RCRdaemon_nersc
#export PAPI_ROOT=/opt/cray/papi/5.3.2.1
#export TAU_ROOT=$HOME/install/tau2-hpx
#export ACTIVEHARMONY_ROOT=$HOME/install/activeharmony/4.5

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
-DUSE_BINUTILS=TRUE \
-DCMAKE_CXX_COMPILER=CC \
" # anything else?

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# run the main script
. $DIR/bootstrap-main.sh
