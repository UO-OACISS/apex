#!/bin/bash -e

# get how many cores we have to use for the build
procs=1
if [ -f '/proc/cpuinfo' ] ; then
  procs=`grep -c ^processor /proc/cpuinfo`
fi

# build PAPI
# ------------------------------------------------------------------------

#configure parameters
version=5.3.2
papi_root=`pwd`/papi-$version

# get the papi tarball
file=papi-$version.tar.gz
if [ ! -f $file ] ; then
	wget http://icl.cs.utk.edu/projects/papi/downloads/$file
fi
tar -xvzf $file

# configure and build
cd papi-$version/src
./configure --prefix=$papi_root
make -j `expr $procs + 1`
make install
cd ../..

echo "PAPI is installed in $papi_root"
