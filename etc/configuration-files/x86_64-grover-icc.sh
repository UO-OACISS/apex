#platform_malloc="-DJEMALLOC_ROOT=${HOME}/install/jemalloc-3.5.1"
platform_malloc="-DGPERFTOOLS_ROOT=${HOME}/install/gperftools/2.5"
platform_bfd="-DBFD_ROOT=${HOME}/install/binutils-2.25-knl"
platform_ah="-DACTIVEHARMONY_ROOT=${HOME}/install/activeharmony/4.6.0-knl"
platform_otf="-DOTF2_ROOT=${HOME}/install/otf2/2.0-knl"
platform_ompt="-DOMPT_ROOT=${HOME}/src/LLVM-openmp/build-icc-Release -DAPEX_WITH_PLUGINS=TRUE"
platform_mpi=""
platform_papi="-DAPEX_WITH_PAPI=TRUE -DPAPI_ROOT=${HOME}/install/papi/5.5.0-knl"
platform_tau="-DAPEX_WITH_TAU=TRUE -DTAU_ROOT=$HOME/src/tau2 -DTAU_ARCH=x86_64 -DTAU_OPTIONS=-icpc-pthread"

module load cmake intel gcc/5.3

parallel_build="-j 32"
