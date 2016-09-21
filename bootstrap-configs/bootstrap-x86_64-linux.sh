#!/bin/bash -e

# where is this script located?
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

###################################################################
#configure parameters - set what ever you need in this top section!
###################################################################

# REQUIRED libraries

BOOST_ROOT=$HOME/hpx/boost_1_57_0

# OPTIONAL libraries - if left undefined, they likely won't be used.

export BFD_ROOT=$HOME/hpx/tau-2.25.1/x86_64/binutils-2.23.2
#export JEMALLOC_ROOT=$HOME/hpx/jemalloc-inst
#export GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#export RCR_ROOT=$HOME/src/RCRdaemon
export PAPI_ROOT=$HOME/hpx/papi-inst
export TAU_ROOT=$HOME/hpx/tau-2.25.1
#export OMPT_ROOT=/home/khuck/src/LLVM-openmp/runtime/build
export ACTIVEHARMONY_ROOT=$HOME/hpx/activeharmony-4.6.0
export OCR_ROOT=$HOME/ocr/ocr/install
export OTF2_ROOT=$HOME/hpx/otf2-2.0-inst

# other CMake variables - for special situations / architectures / compilers.

cmake_build_type="-DCMAKE_BUILD_TYPE=Release" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\"" # if you want to debug in CodeBlocks
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON
cmake_use_mpi="-DUSE_MPI=FALSE" # TRUE or FALSE 
cmake_other_settings="-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_LM_SENSORS=FALSE -DBUILD_EXAMPLES=FALSE -DBUILD_TESTS=FALSE -DTAU_ARCH=x86_64 -DTAU_OPTIONS=-papi-pthread -DUSE_OCR=TRUE -DOCR_ROOT=$OCR_ROOT -DUSE_OTF2=1 -DOTF2_ROOT=$OTF2_ROOT -DUSE_LOAD_BALANCE=TRUE" # anything else?

###################################################################
# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
###################################################################

# ------------------------------------------------------------------------

# run the main script
. $DIR/bootstrap-main.sh
