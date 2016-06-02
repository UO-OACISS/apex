#!/bin/bash -e

# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
# ------------------------------------------------------------------------

boost_config=""
if [ ${BOOST_ROOT+x} ]; then
    boost_config="-DBOOST_ROOT=$BOOST_ROOT"
fi

tau_config=""
if [ ${TAU_ROOT+x} ]; then
    tau_config="-DUSE_TAU=TRUE -DTAU_ROOT=$TAU_ROOT -DTAU_ARCH=$TAU_ARCH -DTAU_OPTIONS=$TAU_OPTIONS"
fi

if [ ${RCR_ROOT+x} ]; then
    rcr_config="-DRCR_ROOT=$RCR_ROOT"
fi

if [ ${SOS_ROOT+x} ]; then
    sos_config="-DSOS_ROOT=$SOS_ROOT"
fi

if [ ${BFD_ROOT+x} ]; then 
    bfd_config="-DBFD_ROOT=$BFD_ROOT -DUSE_BFD=TRUE"
#else
    #bfd_config="-DUSE_BFD=FALSE"
fi

if [ ${JEMALLOC_ROOT+x} ]; then
    jemalloc_config="-DJEMALLOC_ROOT=$JEMALLOC_ROOT -DUSE_JEMALLOC=TRUE"
else
    jemalloc_config=""
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

if [ ${ACTIVEHARMONY_ROOT+x} ]; then
    ah_config="-DACTIVEHARMONY_ROOT=$ACTIVEHARMONY_ROOT -DUSE_ACTIVEHARMONY=TRUE"
elif [ ${HARMONY_HOME+x} ]; then
    ah_config="-DACTIVEHARMONY_ROOT=$HARMONY_HOME -DUSE_ACTIVEHARMONY=TRUE"
else
    ah_config="-DUSE_ACTIVEHARMONY=FALSE"
fi

if [ ${OMPT_ROOT+x} ]; then
    ompt_config="-DOMPT_ROOT=$OMPT_ROOT -DUSE_OMPT=TRUE"
else
    ompt_config="-DUSE_OMPT=FALSE"
fi


# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"

if [ $# -gt 0 ] ; then
    if [ $1 == "--clean" ] || [ $1 == "-c" ] ; then
		if [ x"$WORKDIR" != "x" ] ; then
        	rm -rf ${WORKDIR}/build_*
		else
        	rm -rf build_*
		fi
    fi
fi

datestamp=`date +%Y.%m.%d-%H.%M.%S`
dir=""
if [ x"$WORKDIR" != "x" ] ; then
    dir="${WORKDIR}/build_$datestamp"
else
    dir="build_$datestamp"
fi
mkdir $dir
cd $dir

if [[ $cmake_build_type =~ "Debug" ]] ; then
    export CTEST_OUTPUT_ON_FAILURE=1
fi

cmd="cmake \
$cmake_use_codeblocks \
$boost_config \
$tau_config \
$rcr_config \
$bfd_config \
$jemalloc_config \
$gperftools_config \
$papi_config \
$ah_config \
$ompt_config \
$sos_config \
$cmake_build_type \
$cmake_apex_throttle \
$cmake_build_shared_libs \
$cmake_install_prefix \
$cmake_make_verbose \
$cmake_use_mpi \
$cmake_other_settings \
$DIR/.."
echo $cmd
eval $cmd

procs=0
if [ $# -eq 2 ] ; then
    if [ $2 == "--parallel" ] || [ $2 == "-j" ] ; then
        if [ -f '/proc/cpuinfo' ] ; then
            procs=`grep -c ^processor /proc/cpuinfo`
        fi
        if [ `uname` == "Darwin" ] ; then
            procs=`sysctl -n hw.ncpu`
        fi
    fi
fi

$MAKE_PREFIX make -j `expr $procs + 1`

#make tests -j `expr $procs + 1`
#make examples -j `expr $procs + 1`
make doc
make install

printf "\nSUCCESS!\n"
T="$(($(date +%s)-T))"
printf "Time to configure and build APEX: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
