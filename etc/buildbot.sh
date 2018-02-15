#!/bin/bash
#set -x # echo all commands

my_readlink()
{
    TARGET=$1

    cd $(dirname "$TARGET")
    TARGET=$(basename "$TARGET")

    # Iterate down a (possible) chain of symlinks
    while [ -L "$TARGET" ]
    do
        TARGET=$(readlink "$TARGET")
        cd $(dirname "$TARGET")
        TARGET=$(basename "$TARGET")
    done

    # Compute the canonicalized name by finding the physical path 
    # for the directory we're in and appending the target file.
    DIR=`pwd -P`
    RESULT="$DIR/$TARGET"

    echo $RESULT
}

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
yes_ah=" -DUSE_ACTIVEHARMONY=TRUE -DACTIVEHARMONY_ROOT=/usr/local/activeharmony/4.6 -DUSE_PLUGINS=TRUE"
yes_otf=" -DUSE_OTF2=TRUE -DOTF2_ROOT=/usr/local/otf2/2.1"
yes_ompt=" -DUSE_OMPT=TRUE -DOMPT_ROOT=/usr/local/LLVM-ompt/Release"
#yes_mpi=" -DUSE_MPI=TRUE -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx"
yes_mpi=" -DUSE_MPI=TRUE"
yes_papi=" -DUSE_PAPI=TRUE -DPAPI_ROOT=/usr/local/papi/5.5.0"
yes_tau=" -DUSE_TAU=TRUE -DTAU_ROOT=/usr/local/tau/git -DTAU_ARCH=x86_64 -DTAU_OPTIONS=-pthread"

# set defaults
build="default"
buildtype="Release"
step="config"
dirname="default"
options=""
static=""
sanitize=""
ncores=2
osname=`uname`
if [ ${osname} == "Darwin" ]; then
ncores=`sysctl -n hw.ncpu`
export CC=`which clang`
export CXX=`which clang++`
else
ncores=`nproc --all`
fi

echo "Num parallel builds: $ncores"

set -e # exit on error

# remember where we are
STARTDIR=`pwd`
# Absolute path to this script, e.g. /home/user/bin/foo.sh
if [ ${osname} == "Darwin" ]; then
    SCRIPT=$(my_readlink "$0")
else
    SCRIPT=$(readlink -f "$0")
fi
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH
BASEDIR=$SCRIPTPATH/..

args=$(getopt -l "searchpath:" -o "b:t:s:d:hmn" -- "$@")

eval set -- "${args}"

while [ $# -ge 1 ]; do
    case "$1" in
        --)
            # No more options left.
            break
            ;;
        -b|--build)
            build="$2"
            ;;
        -t|--type)
            buildtype="$2"
            ;;
        -s|--step)
            step="$2"
            ;;
        -d|--dirname)
            dirname="$2"
            ;;
        -m|--sanitize)
            sanitize="-DAPEX_SANITIZE=TRUE"
            ;;
        -n|--static)
            static="-DBUILD_STATIC_EXECUTABLES=TRUE"
            yes_mpi="" # no static MPI installation. :(
            ;;
        -h)
            echo "$0 -b,--build [default|base|malloc|bfd|ah|ompt|papi|mpi|otf|tau] -t,--type [Release|Debug] -s,--step [config|compile|pcompile|test|install] -d,--dirname <dirname> -m,--sanitize -n,--static"
            exit 0
            ;;
    esac
    shift
done

echo "build = ${build}"
echo "buildtype = ${buildtype}"
echo "step = ${step}"
echo "dirname = ${dirname}"

cmake_prefix="cmake .. -DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE="

config_step()
{
    rm -rf ${dirname}
    mkdir -p ${dirname}
    cd ${dirname}
    cmake_cmd="${cmake_prefix}${buildtype} ${options} ${static} ${sanitize}"
    echo ${cmake_cmd}
    ${cmake_cmd}
}

compile_step()
{
    cd ${dirname}
    make ${1}
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

if [ ${build} == "install" ] ; then
    options="${yes_malloc} ${yes_bfd} ${yes_ah} ${yes_ompt} ${yes_papi} ${yes_mpi} ${yes_otf} ${yes_tau}"
    cmake_prefix="cmake .. -DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DCMAKE_INSTALL_PREFIX=${HOME}/install/${buildtype} -DCMAKE_BUILD_TYPE="
    envfile="apex-ah-ompt-papi-mpi-tau.conf"
fi

if [ ${step} == "config" ] ; then
    config_step
fi
if [ ${step} == "compile" ] ; then
    compile_step
fi
if [ ${step} == "pcompile" ] ; then
    compile_step -j${ncores} -l${ncores}
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
