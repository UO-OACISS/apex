#platform_malloc="-DJEMALLOC_ROOT=$HOME/install/jemalloc-3.5.1"
#platform_malloc="-DGPERFTOOLS_ROOT=$HOME/install/gperftools/2.5"
platform_malloc="-DJEMALLOC_ROOT=/usr/local/packages/jemalloc/4.2.1"
platform_bfd="-DBFD_ROOT=$HOME/src/tau2/x86_64/binutils-2.23.2"
platform_ah="-DACTIVEHARMONY_ROOT=$HOME/install/activeharmony/4.6.0"
platform_otf="-DOTF2_ROOT=$HOME/install/otf2/2.0"
platform_ompt="-DOMPT_ROOT=$HOME/src/LLVM-openmp/build-gcc-Release -DUSE_PLUGINS=TRUE"
#platform_mpi="-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_MPI=TRUE"
platform_mpi=""
#platform_papi="-DUSE_PAPI=TRUE -DPAPI_ROOT=${PAPI}"
platform_papi=""
platform_tau="-DUSE_TAU=TRUE -DTAU_ROOT=$HOME/src/tau2 -DTAU_ARCH=x86_64 -DTAU_OPTIONS=-pthread"

module purge
module load cmake 
#mpi-tor/openmpi-1.8_gcc-4.9 papi 
module load gcc/5.3
#module swap gcc gcc/6.1

parallel_build="-j 24"
