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

no_bfd=" -DAPEX_WITH_BFD=FALSE"
no_ah=" -DAPEX_WITH_ACTIVEHARMONY=FALSE"
no_papi=" -DAPEX_WITH_PAPI=FALSE"
no_otf=" -DAPEX_WITH_OTF2=FALSE"
yes_bfd=" -DAPEX_WITH_BFD=TRUE"
yes_ah=" -DAPEX_WITH_ACTIVEHARMONY=TRUE -DACTIVEHARMONY_ROOT=/usr/local/activeharmony/4.6 -DAPEX_WITH_PLUGINS=TRUE"
yes_otf=" -DAPEX_WITH_OTF2=TRUE -DOTF2_ROOT=/usr/local/otf2/2.1"
yes_papi=" -DAPEX_WITH_PAPI=TRUE -DPAPI_ROOT=/usr/local/papi/5.5.0"

# set defaults
build="default"
buildtype="Release"
step="config"
dirname="hpx-build"
options=""
static=""
sanitize=""
ncores=2
osname=`uname`
if [ ${osname} == "Darwin" ]; then
ncores=`sysctl -n hw.ncpu`
export CC=`which clang`
export CXX=`which clang++`
#cmake_generator="-G Xcode"
else
ncores=`nproc --all`
ncores=`expr $ncores / 2`
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
            ;;
        -h)
            echo "$0 -b,--build [default|base|bfd|ah|papi|otf] -t,--type [Release|Debug] -s,--step [config|compile|pcompile|test|install] -d,--dirname <dirname> -m,--sanitize -n,--static"
            exit 0
            ;;
    esac
    shift
done

echo "build = ${build}"
echo "buildtype = ${buildtype}"
echo "step = ${step}"
echo "dirname = ${dirname}"

cmake_prefix="cmake ${cmake_generator} .. -DBUILD_EXAMPLES=TRUE -DBUILD_TESTS=TRUE -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE="

config_step()
{
    rm -rf ${dirname}
    mkdir -p ${dirname}
    cd ${dirname}
    rm -rf /dev/shm/hpx
    git clone --branch stable https://github.com/STEllAR-GROUP/hpx.git /dev/shm/hpx
    mkdir build
    cd build
    cmake_cmd="cmake \
        -DCMAKE_BUILD_TYPE=${buildtype} \
        -DBOOST_ROOT=/usr/local/boost/1.65.0-gcc6 \
        -DTCMALLOC_ROOT=/usr/local/gperftools/2.5 \
        -DHPX_WITH_MALLOC=tcmalloc \
        -DCMAKE_INSTALL_PREFIX=. \
        -DHPX_WITH_THREAD_IDLE_RATES=OFF \
        -DHPX_WITH_PARCELPORT_MPI=ON \
        -DHPX_WITH_PARCEL_COALESCING=ON \
        -DHPX_WITH_TOOLS=OFF \
        -DHPX_WITH_STACKOVERFLOW_DETECTION_DEFAULT=OFF \
        -DHPX_WITH_APEX=TRUE \
        -DHPX_WITH_APEX_NO_UPDATE=TRUE \
        -DHPX_WITH_APEX_TAG=develop \
        ${options} ${static} \
        /dev/shm/hpx"

    #cmake_cmd="${cmake_prefix}${buildtype} ${options} ${static} ${sanitize}"
    echo ${cmake_cmd}
    ${cmake_cmd}
}

compile_step()
{
    cd ${dirname}/build
    make ${1}
}

test_step()
{
    cd ${dirname}/build
    . ../../etc/configuration-files/${envfile}
    env | grep APEX
    export CTEST_OUTPUT_ON_FAILURE=1
    export APEX_SCREEN_OUTPUT=1
    make ${1} tests.examples
    ctest -V --timeout 10 -R tests.examples
}

install_step()
{
    cd ${dirname}
    make doc
    make install
}

if [ ${build} == "default" ] ; then
    options="${no_bfd} ${no_ah} ${no_papi} ${no_otf}"
    envfile="apex-defaults.conf"
fi

if [ ${build} == "base" ] ; then
    options="${no_bfd} ${no_ah} ${no_papi} ${no_otf}"
    envfile="apex-base.conf"
fi

if [ ${build} == "bfd" ] ; then
    options="${yes_bfd} ${no_ah} ${no_papi} ${no_otf}"
    envfile="apex-base.conf"
fi

if [ ${build} == "ah" ] ; then
    options="${yes_bfd} ${yes_ah} ${no_papi} ${no_otf}"
    envfile="apex-ah.conf"
fi

if [ ${build} == "papi" ] ; then
    options="${yes_bfd} ${yes_ah} ${yes_papi} ${no_otf}"
    envfile="apex-ah-ompt-papi.conf"
fi

if [ ${build} == "otf" ] ; then
    options="${yes_bfd} ${yes_ah} ${yes_papi} ${yes_otf}"
    envfile="apex-ah-ompt-papi-mpi-otf.conf"
fi

if [ ${build} == "install" ] ; then
    options="${yes_bfd} ${yes_ah} ${yes_papi} ${yes_otf}"
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
    test_step -j${ncores} -l${ncores}
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
    #install_step
fi
