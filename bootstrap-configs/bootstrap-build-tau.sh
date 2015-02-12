#!/bin/bash -e

papi_root=`pwd`/papi-5.3.2

# NO NEED TO MODIFY ANYTHING BELOW THIS LINE
# ------------------------------------------------------------------------

# get how many cores we have to use for the build
procs=1
if [ -f '/proc/cpuinfo' ] ; then
  procs=`grep -c ^processor /proc/cpuinfo`
fi

# To build TAU
# ------------------------------------------------------------------------

#configure parameters
version=tau2-git-latest
export TAU_ROOT=`pwd`/$version

# get the tarball
file=$version.tar.gz
if [ ! -f $file ] ; then
	wget http://www.nic.uoregon.edu/~khuck/$file
fi
tar -xvzf $file

# configure and build
cd $TAU_ROOT
./configure -pthread -bfd=download -unwind=download -tag=hpx -papi=$papi_root

make -j `expr $procs + 1` install

echo ""
echo "TAU build complete."
echo "Set the TAU_ROOT environment variable to $TAU_ROOT."
echo "Add $TAU_ROOT/`./utils/archfind`/bin to your path."
echo "Add $papi_root/bin to your path."

cd ..
