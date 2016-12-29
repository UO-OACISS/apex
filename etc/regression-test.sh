#!/bin/bash
# exit on error
set -e

sanitize=""
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

args=$(getopt -l "searchpath:" -o "s:chm" -- "$@")

eval set -- "${args}"

while [ $# -ge 1 ]; do
    case "$1" in
        --)
            # No more options left.
            break
            ;;
        -s|--spec)
            spec="$2"
            ;;
        -m)
            sanitize="-DAPEX_SANITIZE=TRUE"
            ;;
        -c)
            clean=1
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
    echo "CC is set to '${CC}'"
    export CC
fi
if [ -z ${CXX+x} ]; then 
    echo "CXX is unset, using gcc"
    export CXX=g++
else 
    echo "CXX is set to '${CXX}'"
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

    rm -rf build${post}-${buildtype} ${BASEDIR}/install${post}-${buildtype}
    mkdir build${post}-${buildtype}
    cd build${post}-${buildtype}
    cmd="cmake -DCMAKE_BUILD_TYPE=${buildtype} -DBUILD_TESTS=TRUE ${sanitize} \
    -DBUILD_EXAMPLES=TRUE ${malloc} ${bfd} ${ah} ${ompt} ${papi} ${mpi} ${otf} ${tau} ${extra} \
    -DCMAKE_INSTALL_PREFIX=${BASEDIR}/install${post}-${buildtype} ${BASEDIR}"
    echo ${cmd}
    ${cmd} 2>&1 | tee -a ${logfile}
    make ${parallel_build} 2>&1 | tee -a ${logfile}
    make doc 2>&1 | tee -a ${logfile}
    make install 2>&1 | tee -a ${logfile}
    #make test 2>&1 | tee -a ${logfile}
    #ctest --repeat-until-fail 5 --output-on-failure 2>&1 | tee -a ${logfile}
    ctest --output-on-failure 2>&1 | tee -a ${logfile}
    printf "\nSUCCESS!\n"
    T="$(($(date +%s)-T))"
    printf "Time to configure and build APEX: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
    cd ..
}

conditional_build()
{
    echo >&2 "Sourcing ${configfile}"
    source ${configfile}
    echo "spec: ${spec}"
    echo "post: ${post}"
    if [ "${spec}" = "all" ] ; then
        buildtype=Debug
        #extra="-DAPEX_DEBUG=TRUE"
        extra=""
        dobuild
        buildtype=Release
        extra=""
        dobuild
    elif [ "${post}" = "" ] ; then
        return
    else
        if [[ "${post}" == *"${spec}" ]] ; then
            buildtype=Debug
            #extra="-DAPEX_DEBUG=TRUE"
            extra=""
            dobuild
            buildtype=Release
            extra=""
            dobuild
        fi
    fi
}

# Clean settings
malloc=""
bfd="-DUSE_BFD=FALSE"
ah="-DUSE_ACTIVEHARMONY=FALSE"
ompt="-DUSE_OMPT=FALSE"
papi="-DUSE_PAPI=FALSE"
mpi="-DUSE_MPI=FALSE"
otf="-DUSE_OTF2=FALSE"
tau="-DUSE_TAU=FALSE"

if [ ${clean} -eq 1 ] ; then
    echo "cleaning previous regression test..."
    rm -rf /dev/shm/regression-${host}
    mkdir -p /dev/shm/regression-${host}
    git checkout develop
    git pull
fi

# change directory to the base APEX directory
cd /dev/shm/regression-${host}

logfile=`pwd`/log.txt
configfile=${SCRIPTPATH}/configuration-files/apex-defaults.conf
conditional_build

post=-base
configfile=${SCRIPTPATH}/configuration-files/apex-base.conf
conditional_build

malloc=${platform_malloc}
post=-malloc
configfile=${SCRIPTPATH}/configuration-files/apex-base.conf
conditional_build

bfd=${platform_bfd}
post=${post}-bfd
configfile=${SCRIPTPATH}/configuration-files/apex-base.conf
conditional_build

ah=${platform_ah}
post=${post}-ah
configfile=${SCRIPTPATH}/configuration-files/apex-ah.conf
conditional_build

ompt=${platform_ompt}
post=${post}-ompt
configfile=${SCRIPTPATH}/configuration-files/apex-ah-ompt.conf
conditional_build

papi=${platform_papi}
post=${post}-papi
configfile=${SCRIPTPATH}/configuration-files/apex-ah-ompt-papi.conf
conditional_build

mpi=${platform_mpi}
post=${post}-mpi
configfile=${SCRIPTPATH}/configuration-files/apex-ah-ompt-papi-mpi.conf
conditional_build

otf=${platform_otf}
post=${post}-otf
configfile=${SCRIPTPATH}/configuration-files/apex-ah-ompt-papi-mpi-otf.conf
conditional_build

tau=${platform_tau}
post=${post}-tau
configfile=${SCRIPTPATH}/configuration-files/apex-ah-ompt-papi-mpi-tau.conf
conditional_build

cd ${STARTDIR}