#!/bin/bash
# exit on error
set -e

clean=0
spec="all"
post=""
host=`hostname`
myarch=`arch`

# remember where we are
STARTDIR=`pwd`
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH
BASEDIR=$SCRIPTPATH/..

args=$(getopt -l "searchpath:" -o "s:ch" -- "$@")

eval set -- "${args}"

while [ $# -ge 1 ]; do
    case "$1" in
        --)
            # No more options left.
            shift
            break
            ;;
        -s|--spec)
            spec="$2"
            shift
            ;;
        -c)
            clean=1
            shift
            ;;
        -h)
            echo "Display some help"
            exit 0
            ;;
    esac

    shift
done

echo "clean: ${clean}"
echo "spec: ${spec}"
echo "hostname: ${host}"
echo "remaining args: $*"
if [ -z ${CC+x} ]; then 
	echo "CC is unset, using gcc"
	export CC=gcc
else 
	echo "CC is set to '$var'"
	export CC
fi
if [ -z ${CXX+x} ]; then 
	echo "CXX is unset, using gcc"
	export CXX=g++
else 
	echo "CXX is set to '$var'"
	export CXX
fi

# source our environment
configfile=${SCRIPTPATH}/configuration-files/${myarch}-${host}-${CC}.sh
echo >&2 "Sourcing ${configfile}"
source ${configfile}

command -v cmake >/dev/null 2>&1 || { 
    echo >&2 "I require cmake but...it ain't there. Check your environment."; 
	return
}

dobuild()
{
    # Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
    T="$(date +%s)"

    rm -rf build${post} install${post}
    mkdir build${post}
    cd build${post}
    cmd="cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=TRUE \
    -DBUILD_EXAMPLES=TRUE $malloc $bfd $ah $otf $ompt $tau $mpi \
    -DCMAKE_INSTALL_PREFIX=../install${post} ../.."
    echo $cmd
    $cmd >> $logfile 2>&1 
    make -j 24 >> $logfile 2>&1 
    make doc >> $logfile 2>&1 
    make install >> $logfile 2>&1 
    make test >> $logfile 2>&1 
    printf "\nSUCCESS!\n"
    T="$(($(date +%s)-T))"
    printf "Time to configure and build APEX: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
    cd ..
}

conditional_build()
{
	echo "spec: ${spec}"
	echo "post: ${post}"
	if [ "${spec}" = "all" ] ; then
		dobuild
	elif [ "${post}" = "" ] ; then
		return
	else
		if [[ "${post}" == *"${spec}" ]] ; then
			dobuild
		fi
	fi
}

# Clean settings
malloc=""
bfd="-DUSE_BFD=FALSE"
ah="-DUSE_ACTIVEHARMONY=FALSE"
ompt="-DUSE_OMPT=FALSE"
otf="-DUSE_OTF2=FALSE"
tau="-DUSE_TAU=FALSE"
mpi="-DUSE_MPI=FALSE"
papi="-DUSE_PAPI=FALSE"

if [ ${clean} -eq 1 ] ; then
    echo "cleaning previous regression test..."
    rm -rf ${BASEDIR}/regression-${host}
    mkdir -p ${BASEDIR}/regression-${host}
	git checkout develop
	git pull
fi

# change directory to the base APEX directory
cd ${BASEDIR}/regression-${host}

logfile=`pwd`/log.txt
configfile=${SCRIPTPATH}/configuration-files/apex-defaults.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

#malloc="-DJEMALLOC_ROOT=$HOME/install/jemalloc-3.5.1"
malloc=${platform_malloc}
post=-malloc
configfile=${SCRIPTPATH}/configuration-files/apex-base.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

bfd=${platform_bfd}
post=${post}-bfd
conditional_build

ah=${platform_ah}
post=${post}-ah
configfile=${SCRIPTPATH}/configuration-files/apex-ah.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

otf=${platform_otf}
post=${post}-otf
configfile=${SCRIPTPATH}/configuration-files/apex-ah-otf.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

ompt=${platform_ompt}
post=${post}-ompt
configfile=${SCRIPTPATH}/configuration-files/apex-ah-otf-ompt.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

mpi=${platform_mpi}
post=${post}-mpi
configfile=${SCRIPTPATH}/configuration-files/apex-ah-otf-ompt-mpi.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

papi=${platform_papi}
post=${post}-papi
configfile=${SCRIPTPATH}/configuration-files/apex-ah-otf-ompt-mpi-papi.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

tau=${platform_tau}
post=${post}-tau
configfile=${SCRIPTPATH}/configuration-files/apex-ah-otf-ompt-mpi-papi-tau.conf
echo >&2 "Sourcing ${configfile}"
source ${configfile}
conditional_build

cd ${STARTDIR}