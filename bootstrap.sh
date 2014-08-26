#!/bin/bash -e

#configure parameters
export TAU_ROOT=/usr/local/tau/git
export BOOST_ROOT=/usr
export HPX_HAVE_ITTNOTIFY=1

# runtime parameters
export APEX_POLICY=1
export APEX_TAU=1
export HPX_HAVE_ITTNOTIFY=1

datestamp=`date +%Y.%m.%d-%H.%M.%S`
mkdir build-$datestamp
cd build-$datestamp

cmake \
-DBOOST_ROOT=$BOOST_ROOT \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_INSTALL_PREFIX=. \
-DHPX_HAVE_APEX=1 \
-DTAU_ROOT=$TAU_ROOT \
..

make -j
make install
make test
