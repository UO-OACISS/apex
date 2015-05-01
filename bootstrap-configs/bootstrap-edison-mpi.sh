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
export JEMALLOC_ROOT=$HOME/install/jemalloc/3.6.0 # not strictly necessary, if runtime uses JEMalloc
#export RCR_ROOT=$HOME/src/RCRdaemon
#export PAPI_ROOT=/usr/local/papi/5.3.2
#export TAU_ROOT=$HOME/src/tau2-edison

# other CMake variables - for special situations / architectures / compilers.

# For edison: CMake will get these from the environment
export LDFLAGS="-dynamic -ldl" # cmake will pick this up
export MPI_C_INCLUDE_PATH=$CRAY_MPICH2_DIR/include
export MPI_C_LIBRARIES="-L$CRAY_MPICH2_DIR/lib -lmpi"
cmake_build_type="-DCMAKE_BUILD_TYPE=RelWithDebInfo" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=FALSE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=TRUE" # TRUE or FALSE
cmake_other_settings="\
-DMPI_C_INCLUDE_PATH=$MPI_C_INCLUDE_PATH \
-DMPI_CXX_INCLUDE_PATH=$MPI_C_INCLUDE_PATH \
-DMPI_C_LIBRARIES=$MPI_C_LIBRARIES \
-DMPI_CXX_LIBRARIES=$MPI_C_LIBRARIES \
-DMPI_C_COMPILER=cc \
-DMPI_CXX_COMPILER=CC \
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
