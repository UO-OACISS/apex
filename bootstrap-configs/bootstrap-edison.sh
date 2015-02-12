#!/bin/bash -e

#configure parameters
export BOOST_ROOT=$BOOST_DIR
export LDFLAGS="-dynamic -ldl"
#export CFLAGS=-fPIC
#export CXXFLAGS=-fPIC
#export BOOST_ROOT=$HOME/install/boost-1.56.0
#export RCR_ROOT=$HOME/src/RCRdaemon
#export PAPI_ROOT=/usr/local/papi/5.3.2
#export TAU_ROOT=$HOME/src/tau2
#export BFD_ROOT=$HOME/src/tau2/x86_64/binutils-2.23.2
# this one is only meaningful for HPX-3 from LSU
# export HPX_HAVE_ITTNOTIFY=1 

# For edison:
export MPI_C_INCLUDE_PATH=$CRAY_MPICH2_DIR/include
export MPI_C_LIBRARIES="-L$CRAY_MPICH2_DIR/lib -lmpi"

# runtime parameters for HPX-3 (LSU)
export APEX_POLICY=1
export APEX_CONCURRENCY=0
export APEX_TAU=0
# this one is only meaningful for HPX-3 from LSU
# export HPX_HAVE_ITTNOTIFY=1

# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
# ------------------------------------------------------------------------

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"

if [ $# -eq 1 ] ; then
	if [ $1 == "--clean" ] || [ $1 == "-c" ] ; then
		rm -rf build_*
	fi
fi

datestamp=`date +%Y.%m.%d-%H.%M.%S`
dir="build_$datestamp"
mkdir $dir
cd $dir

# Enable shared libraries, if desired.
#-DTAU_ROOT=$TAU_ROOT \
#-G "CodeBlocks - Unix Makefiles" \
#-DCMAKE_EXE_LINKER_FLAGS=$LDFLAGS \
#-DCMAKE_C_FLAGS=$CFLAGS \
#-DCMAKE_CXX_FLAGS=$CXXFLAGS \
#-DCMAKE_VERBOSE_MAKEFILE=ON \

cmake \
-DBOOST_ROOT=$BOOST_ROOT \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DAPEX_THROTTLE=FALSE \
-DBUILD_SHARED_LIBS=FALSE \
-DUSE_BINUTILS=TRUE \
-DCMAKE_INSTALL_PREFIX=../install \
-DMPI_C_INCLUDE_PATH=$MPI_C_INCLUDE_PATH \
-DMPI_CXX_INCLUDE_PATH=$MPI_C_INCLUDE_PATH \
-DMPI_C_LIBRARIES=$MPI_C_LIBRARIES \
-DMPI_CXX_LIBRARIES=$MPI_C_LIBRARIES \
-DMPI_C_COMPILER=cc \
-DMPI_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_CXX_COMPILER=CC \
..

procs=1
if [ -f '/proc/cpuinfo' ] ; then
  procs=`grep -c ^processor /proc/cpuinfo`
fi
make -j `expr $procs + 1`

# make test
make doc
make install

printf "\nSUCCESS!\n"
T="$(($(date +%s)-T))"
printf "Time to configure and build APEX: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
