#!/bin/bash
#set -x # echo all commands
set -e # exit on error

no_malloc=""
no_bfd=" -DUSE_BFD=FALSE"
no_ah=" -DUSE_ACTIVEHARMONY=FALSE"
no_ompt=" -DUSE_OMPT=FALSE"
no_papi=" -DUSE_PAPI=FALSE"
no_mpi=" -DUSE_MPI=FALSE"
no_otf=" -DUSE_OTF2=FALSE"
no_tau=" -DUSE_TAU=FALSE"
yes_malloc=" -DUSE_JEMALLOC=TRUE"
yes_bfd=" -DUSE_BFD=TRUE"
yes_ah=" -DUSE_ACTIVEHARMONY=TRUE -DACTIVEHARMONY_ROOT=/usr/local/activeharmony/4.6"
yes_otf=" -DUSE_OTF2=TRUE -DOTF2_ROOT=/usr/local/otf2/2.0"
yes_ompt=" -DUSE_OMPT=TRUE -DOMPT_ROOT=/usr/local/LLVM-ompt/Release"
yes_mpi=" -DUSE_MPI=TRUE -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx"
yes_papi=" -DUSE_PAPI=TRUE -DPAPI_ROOT=/usr/local/papi/5.5.0"
yes_tau=" -DUSE_TAU=TRUE -DTAU_ROOT=/usr/local/tau/git -DTAU_ARCH=x86_64 -DTAU_OPTIONS=-pthread"

# set defaults
build="default"
buildtype="Release"
step="config"
dirname="default"
options=""

# remember where we are
STARTDIR=`pwd`
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH
BASEDIR=$SCRIPTPATH/..

args=$(getopt -l "searchpath:" -o "b:t:s:d:h" -- "$@")

eval set -- "${args}"

while [ $# -ge 1 ]; do
    case "$1" in
        --)
            # No more options left.
            shift
            break
            ;;
        -b|--build)
            build="$2"
            shift
            ;;
        -t|--type)
            buildtype="$2"
            shift
            ;;
        -s|--step)
            step="$2"
            shift
            ;;
        -d|--dirname)
            dirname="$2"
            shift
            ;;
        -h)
            echo "Display some help"
            exit 0
            ;;
    esac

    shift
done

echo "build = ${build}"
echo "buildtype = ${buildtype}"
echo "step = ${step}"
echo "dirname = ${dirname}"

config_step()
{
    rm -rf ${dirname}
    mkdir -p ${dirname}
    cd ${dirname}
    cmake_cmd="cmake .. -DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=${buildtype} ${options}"
    ${cmake_cmd}
}

compile_step()
{
    cd ${dirname}
    make -j8
}

test_step()
{
    cd ${dirname}
    . ../etc/configuration-files/${envfile}
    env | grep APEX
    export CTEST_OUTPUT_ON_FAILURE=1
    make test
}

install_step()
{
    cd ${dirname}
    make doc
    make install
}

if [ ${build} == "default" ] ; then
    options="${no_malloc} ${no_bfd} ${no_ah} ${no_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-defaults.conf"
fi

if [ ${build} == "base" ] ; then
    options="${no_malloc} ${no_bfd} ${no_ah} ${no_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-base.conf"
fi

if [ ${build} == "malloc" ] ; then
    options="${yes_malloc} ${no_bfd} ${no_ah} ${no_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-base.conf"
fi

if [ ${build} == "bfd" ] ; then
    options="${yes_malloc} ${yes_bfd} ${no_ah} ${no_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-base.conf"
fi

if [ ${build} == "ah" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${no_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-ah.conf"
fi

if [ ${build} == "ompt" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${no_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-ah-ompt.conf"
fi

if [ ${build} == "papi" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${yes_papi} ${no_mpi} ${no_otf} ${no_tau}"
    envfile="apex-ah-ompt-papi.conf"
fi

if [ ${build} == "mpi" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${yes_papi} ${yes_mpi} ${no_otf} ${no_tau}"
    envfile="apex-ah-ompt-papi-mpi.conf"
fi

if [ ${build} == "otf" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${yes_papi} ${yes_mpi} ${yes_otf} ${no_tau}"
    envfile="apex-ah-ompt-papi-mpi-otf.conf"
fi

if [ ${build} == "tau" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${yes_papi} ${yes_mpi} ${yes_otf} ${yes_tau}"
    envfile="apex-ah-ompt-papi-mpi-tau.conf"
fi

if [ ${step} == "config" ] ; then
    config_step
fi
if [ ${step} == "compile" ] ; then
    compile_step
fi
if [ ${step} == "test" ] ; then
    test_step
fi
if [ ${step} == "install" ] ; then
    install_step
fi
if [ ${step} == "all" ] ; then
    config_step
    cd ..
    compile_step
    cd ..
    test_step
    cd ..
    install_step
fi
