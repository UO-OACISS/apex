#platform_malloc="-DJEMALLOC_ROOT=$HOME/install/jemalloc-3.5.1"
platform_malloc="-DGPERFTOOLS_ROOT=/usr/local/packages/gperftools/2.5"
platform_bfd="-DBFD_ROOT=/usr/local/packages/binutils/2.25"
platform_ah="-DACTIVEHARMONY_ROOT=/usr/local/packages/activeharmony/4.6.0-ppc64le"
platform_ompt="-DOMPT_ROOT=/usr/local/packages/LLVM-ompt/power8_gcc-4.9.2 -DUSE_PLUGINS=TRUE"
platform_papi="-DUSE_PAPI=TRUE -DPAPI_ROOT=/usr/local/packages/papi/5.5.0-ppc64le"
#platform_mpi="-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_MPI=TRUE"
platform_mpi="-DUSE_MPI=FALSE"
platform_otf="-DOTF2_ROOT=/usr/local/packages/otf2/2.0-ppc64le"
platform_tau="-DUSE_TAU=TRUE -DTAU_ROOT=$HOME/src/tau2 -DTAU_ARCH=ibm64linux -DTAU_OPTIONS=-pthread"

module load gcc/5.3

parallel_build="-j 20"
