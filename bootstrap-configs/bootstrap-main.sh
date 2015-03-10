#!/bin/bash -e

# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
# ------------------------------------------------------------------------

boost_config=""
if [ ${BOOST_ROOT+x} ]; then
	boost_config="-DBOOST_ROOT=$BOOST_ROOT"
fi

if [ ${RCR_ROOT+x} ]; then
	rcr_config="-DRCR_ROOT=$RCR_ROOT"
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

if [ ${ACTIVEHARMONY_ROOT+x} ]; then
	papi_config="-DACTIVEHARMONY_ROOT=$ACTIVEHARMONY_ROOT -DUSE_ACTIVEHARMONY=TRUE"
elif [ ${HARMONY_HOME+x} ]; then
	papi_config="-DACTIVEHARMONY_ROOT=$HARMONY_HOME -DUSE_ACTIVEHARMONY=TRUE"
else
	papi_config="-DUSE_ACTIVEHARMONY=FALSE"
fi

if [ ${OMPT_ROOT+x} ]; then
	ompt_config="-DOMPT_ROOT=$OMPT_ROOT"
else
	ompt_config="-DUSE_OMPT=FALSE"
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

cmd="cmake \
$cmake_use_codeblocks \
$boost_config \
$rcr_config \
$bfd_config \
$gperftools_config \
$papi_config \
$ompt_config \
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
