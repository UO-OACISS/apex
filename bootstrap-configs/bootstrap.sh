#!/bin/bash -e

#configure parameters - set what ever you need in this top section!

# REQUIRED libraries

BOOST_ROOT=/usr

# OPTIONAL libraries

BFD_ROOT=$HOME/install/binutils-2.23.2
GPERFTOOLS_ROOT=$HOME/install/google-perftools/2.4
#RCR_ROOT=$HOME/src/RCRdaemon
#PAPI_ROOT=/usr/local/papi/5.3.2
#TAU_ROOT=$HOME/src/tau2

# other CMake variables

cmake_build_type="-DCMAKE_BUILD_TYPE=Debug" # Debug, Release, RelWithDebInfo, etc.
cmake_apex_throttle="-DAPEX_THROTTLE=FALSE" # TRUE or FALSE
cmake_build_shared_libs="-DBUILD_SHARED_LIBS=TRUE" # TRUE or FALSE
cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=../install" # the installation path
cmake_use_codeblocks="-G \"CodeBlocks - Unix Makefiles\""
cmake_make_verbose=""  # for verbose, use -DCMAKE_VERBOSE_MAKEFILE=ON

# runtime parameters for testing HPX

export APEX_POLICY=1
export APEX_CONCURRENCY=0
export APEX_TAU=0

# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
# ------------------------------------------------------------------------

boost_config=""
if [ ${BOOST_ROOT+x} ]; then
	boost_config="-DBOOST_ROOT=$BOOST_ROOT"
fi

if [ ${BFD_ROOT+x} ]; then 
	bfd_config="-DBFD_ROOT=$BFD_ROOT -DUSE_BINUTILS=TRUE"
else
	bfd_config="-DUSE_BINUTILS=FALSE"
fi

if [ ${GPERFTOOLS_ROOT+x} ]; then
	gperftools_config="-DGPERFTOOLS_ROOT=$GPERFTOOLS_ROOT"
else
	gperftools_config=""
fi

if [ ${PAPI_ROOT+x} ]; then
	papi_config="-DPAPI_ROOT=$PAPI_ROOT -DUSE_PAPI=TRUE"
else
	papi_config="-DUSE_PAPI=FALSE"
fi


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

cmd="cmake $cmake_use_codeblocks $boost_config $bfd_config $gperftools_config $cmake_build_type $cmake_apex_throttle $cmake_build_shared_libs $cmake_install_prefix $cmake_make_verbose .."
echo $cmd
eval $cmd

procs=1
if [ -f '/proc/cpuinfo' ] ; then
  procs=`grep -c ^processor /proc/cpuinfo`
fi
make -j `expr $procs + 1`

make test
make doc
make install

printf "\nSUCCESS!\n"
T="$(($(date +%s)-T))"
printf "Time to configure and build APEX: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
