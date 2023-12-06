# The name of our project is "APEX". CMakeLists files in this project can
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and
# to the root binary directory of the project as ${APEX_BINARY_DIR}.

# All CMAKE options for the APEX project...
option (APEX_BUILD_TESTS "Build APEX tests (for 'make test')" FALSE)
option (APEX_BUILD_EXAMPLES "Build APEX examples" FALSE)
option (APEX_BUILD_ACTIVEHARMONY "Build ActiveHarmony library if not found" FALSE)
option (APEX_BUILD_BFD "Build Binutils library if not found" FALSE)
#option (APEX_BUILD_OMPT "Build OpenMP runtime with OMPT if support not found" FALSE)
option (APEX_BUILD_OTF2 "Build OTF2 library if not found" FALSE)
option (APEX_CUDA_TESTS "Build APEX CUDA tests (for 'make test')" FALSE)
option (APEX_HIP_TESTS "Build APEX HIP tests (for 'make test')" FALSE)
option (APEX_SANITIZE "Enable compiler sanitizer flags" FALSE)
option (APEX_WITH_ACTIVEHARMONY "Enable ActiveHarmony support" FALSE)
option (APEX_WITH_BFD "Enable Binutils (BFD)support" FALSE)
option (APEX_WITH_CUDA "Enable CUDA (CUPTI) support" FALSE)
option (APEX_WITH_HIP "Enable HIP (ROCTRACER) support" FALSE)
option (APEX_WITH_KOKKOS "Enable Kokkos support" TRUE)
option (APEX_WITH_JEMALLOC "Enable JEMalloc heap management" FALSE)
option (APEX_WITH_LEVEL0 "Enable LEVEL0 (Intel OneAPI) support" FALSE)
option (APEX_WITH_LM_SENSORS "Enable LM Sensors support" FALSE)
option (APEX_WITH_MPI "Enable MPI support" FALSE)
option (APEX_WITH_OMPT "Enable OpenMP Tools (OMPT) support" FALSE)
option (APEX_WITH_OTF2 "Enable Open Trace Format 2 (OTF2) support" FALSE)
option (APEX_WITH_PAPI "Enable PAPI support" FALSE)
option (APEX_WITH_PERFETTO "Enable native Perfetto trace support" FALSE)
option (APEX_WITH_PHIPROF "Enable APEX PhiProf support" FALSE)
option (APEX_WITH_PLUGINS "Enable APEX policy plugin support" FALSE)
option (APEX_WITH_STARPU "Enable APEX StarPU support" FALSE)
option (APEX_WITH_TCMALLOC "Enable TCMalloc heap management" FALSE)
option (APEX_USE_PEDANTIC "Enable pedantic compiler flags" FALSE)
option (APEX_USE_STATIC_GLOBAL_CONSTRUCTOR
    "Enable static global constructor for APEX library load (when __libc_start_main can't be wrapped)" FALSE)

# Provide some backwards compatability
if(DEFINED USE_ACTIVEHARMONY)
    message(WARNING "USE_ACTIVEHARMONY is deprecated - please use APEX_WITH_ACTIVEHARMONY")
    set(APEX_WITH_ACTIVEHARMONY CACHE BOOL ${USE_ACTIVEHARMONY})
endif()

# Provide some backwards compatability
if(DEFINED USE_BFD)
    message(WARNING "USE_BFD is deprecated - please use APEX_WITH_BFD")
    set(APEX_WITH_BFD CACHE BOOL ${USE_BFD})
endif()

# Provide some backwards compatability
if(DEFINED USE_MPI)
    message(WARNING "USE_MPI is deprecated - please use APEX_WITH_MPI")
    set(APEX_WITH_MPI CACHE BOOL ${USE_MPI})
endif()

# Provide some backwards compatability
if(DEFINED USE_OMPT)
    message(WARNING "USE_OMPT is deprecated - please use APEX_WITH_OMPT")
    set(APEX_WITH_OMPT CACHE BOOL ${USE_OMPT})
endif()

# Provide some backwards compatability
if(DEFINED USE_PAPI)
    message(WARNING "USE_PAPI is deprecated - please use APEX_WITH_PAPI")
    set(APEX_WITH_PAPI CACHE BOOL ${USE_PAPI})
endif()

# Provide some backwards compatability
if(DEFINED USE_OTF2)
    message(WARNING "USE_OTF2 is deprecated - please use APEX_WITH_OTF2")
    set(APEX_WITH_OTF2 CACHE BOOL ${USE_OTF2})
endif()

# Provide some backwards compatability
if(DEFINED USE_PLUGINS)
    message(WARNING "USE_PLUGINS is deprecated - please use APEX_WITH_PLUGINS")
    set(APEX_WITH_PLUGINS CACHE BOOL ${USE_PLUGINS})
endif()

# Provide some backwards compatability
if(DEFINED USE_TCMALLOC)
    message(WARNING "USE_TCMALLOC is deprecated - please use APEX_WITH_TCMALLOC")
    set(APEX_WITH_TCMALLOC CACHE BOOL ${USE_TCMALLOC})
endif()

# Provide some backwards compatability
if(DEFINED USE_LM_SENSORS)
    message(WARNING "USE_LM_SENSORS is deprecated - please use APEX_WITH_LM_SENSORS")
    set(APEX_WITH_LM_SENSORS CACHE BOOL ${USE_LM_SENSORS})
endif()

# Provide some backwards compatability
if(DEFINED USE_JEMALLOC)
    message(WARNING "USE_JEMALLOC is deprecated - please use APEX_WITH_JEMALLOC")
    set(APEX_WITH_JEMALLOC CACHE BOOL ${USE_JEMALLOC})
endif()

# Provide some backwards compatability
if(DEFINED BUILD_TESTS)
    message(WARNING "BUILD_TESTS is deprecated - please use APEX_BUILD_TESTS")
    set(APEX_BUILD_TESTS CACHE BOOL ${BUILD_TESTS})
endif()

# Provide some backwards compatability
if(DEFINED BUILD_EXAMPLES)
    message(WARNING "BUILD_EXAMPLES is deprecated - please use APEX_BUILD_EXAMPLES")
    set(APEX_BUILD_EXAMPLES CACHE BOOL ${BUILD_EXAMPLES})
endif()

# Provide some backwards compatability
if(DEFINED BUILD_ACTIVEHARMONY)
    message(WARNING "BUILD_ACTIVEHARMONY is deprecated - please use APEX_BUILD_ACTIVEHARMONY")
    set(APEX_BUILD_ACTIVEHARMONY CACHE BOOL ${BUILD_ACTIVEHARMONY})
endif()

# Provide some backwards compatability
if(DEFINED BUILD_BFD)
    message(WARNING "BUILD_BFD is deprecated - please use APEX_BUILD_BFD")
    set(APEX_BUILD_BFD CACHE BOOL ${BUILD_BFD})
endif()

# Provide some backwards compatability
if(DEFINED BUILD_OMPT OR DEFINED APEX_BUILD_OMPT)
    message(FATAL_ERROR "APEX_BUILD_OMPT is disabled. If OpenMP support is needed, use a compiler with compliant Tool support.")
endif()

# Provide some backwards compatability
if(DEFINED BUILD_OTF2)
    message(WARNING "BUILD_OTF2 is deprecated - please use APEX_BUILD_OTF2")
    set(APEX_BUILD_OTF2 CACHE BOOL ${BUILD_OTF2})
endif()


