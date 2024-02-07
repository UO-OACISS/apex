# Build the OMPT support with the LLVM library.

# This if statement is specific to OMPT, and should not be copied into other
# Find cmake scripts.

if(NOT OMPT_ROOT AND NOT $ENV{OMPT_ROOT} STREQUAL "")
  set(OMPT_ROOT $ENV{OMPT_ROOT})
endif()

pkg_check_modules(PC_OMPT QUIET OMPT)
set(OMPT_DEFINITIONS ${PC_OMPT_CFLAGS_OTHER})

find_path(OMPT_INCLUDE_DIR omp-tools.h
          HINTS ${PC_OMPT_INCLUDEDIR} ${PC_OMPT_INCLUDE_DIRS} ${OMPT_ROOT}/include ${CMAKE_INSTALL_PREFIX}/ompt/include )

find_library(OMPT_LIBRARY NAMES omp iomp5 gomp
             HINTS ${PC_OMPT_LIBDIR} ${PC_OMPT_LIBRARY_DIRS} ${OMPT_ROOT}/lib
			 ${OMPT_ROOT}/lib/* ${CMAKE_INSTALL_PREFIX}/ompt/lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OMPT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OMPT  DEFAULT_MSG
                                  OMPT_LIBRARY OMPT_INCLUDE_DIR)

mark_as_advanced(OMPT_INCLUDE_DIR OMPT_LIBRARY)

# CUDA 10 doesn't work with GCC  9+ so disable target offload in that case
# CUDA 11 doesn't work with GCC 10+ so disable target offload in that case
set(APEX_OMPT_EXTRA_CONFIG "")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10 AND
        CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11)
        message(ERROR " Disabling OpenMP target offload for GCC 10+")
        set(APEX_OMPT_EXTRA_CONFIG "-DOPENMP_ENABLE_LIBOMPTARGET=FALSE")
    else()
        if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
            message(ERROR " Disabling OpenMP target offload for GCC 9+")
            set(APEX_OMPT_EXTRA_CONFIG "-DOPENMP_ENABLE_LIBOMPTARGET=FALSE")
        endif()
    endif()
endif()

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(APEX_BUILD_OMPT OR (NOT OMPT_FOUND))
  set(OMPT_ROOT ${CMAKE_INSTALL_PREFIX}/ompt CACHE STRING "OMPT Root directory")
  message("Attention: Downloading and Building OMPT as external project!")
  message(INFO " A working internet connection is required!")
  include(ExternalProject)
# The GCC 9 compiler added new GOMP functions, so use an archive of the LLVM 12 compiler runtime.
  ExternalProject_Add(project_ompt
    #URL http://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist/LLVM-openmp-2021-05-14.tar.gz
    URL http://tau.uoregon.edu/LLVM-openmp-2021-05-14.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/LLVM-ompt-5.0
    CONFIGURE_COMMAND cmake -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${OMPT_ROOT} -DCMAKE_BUILD_TYPE=Release ${APEX_OMPT_EXTRA_CONFIG} ../project_ompt
    BUILD_COMMAND make libomp-needed-headers all -j${MAKEJOBS}
    INSTALL_COMMAND make install
    INSTALL_DIR ${OMPT_ROOT}
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  #ExternalProject_Get_Property(project_ompt install_dir)
  add_library(omp SHARED IMPORTED)
  set_property(TARGET omp PROPERTY IMPORTED_LOCATION ${OMPT_ROOT}/lib/libomp.so)
  set(OMPT_INCLUDE_DIR "${OMPT_ROOT}/include")
  set(OMPT_LIBRARY "${OMPT_ROOT}/lib/libomp.so")
  # handle the QUIETLY and REQUIRED arguments and set OMPT_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(OMPT  DEFAULT_MSG
                                    OMPT_LIBRARY OMPT_INCLUDE_DIR)
  set(OMPT_FOUND TRUE)
else()
  add_custom_target(project_ompt)
endif()

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

