# - Try to find LibOMPT
# Once done this will define
#  OMPT_FOUND - System has OMPT
#  OMPT_INCLUDE_DIRS - The OMPT include directories
#  OMPT_LIBRARIES - The libraries needed to use OMPT
#  OMPT_DEFINITIONS - Compiler switches required for using OMPT

find_package(PkgConfig)

# First, check if the compiler has built-in support for OpenMP 5.0:
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-fopenmp HAVE_OPENMP)
try_compile(APEX_HAVE_OMPT_NATIVE ${CMAKE_CURRENT_BINARY_DIR}/ompt_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/ompt_test ompt_test ompt_test
    CMAKE_FLAGS -DCMAKE_CXX_FLAGS:STRING="${OpenMP_CXX_FLAGS}"
    -DCMAKE_CMAKE_EXE_LINKER_FLAGS:STRING="${OpenMP_CXX_FLAGS}")

if(${APEX_HAVE_OMPT_NATIVE})
    set(OMPT_FOUND TRUE)
    mark_as_advanced(OMPT_INCLUDE_DIR OMPT_LIBRARY)
    add_custom_target(project_ompt)
    message("Detected compiler has native OpenMP 5.0 OMPT support.")
else(${APEX_HAVE_OMPT_NATIVE})

# This if statement is specific to OMPT, and should not be copied into other
# Find cmake scripts.

if(NOT OMPT_ROOT AND NOT $ENV{OMPT_ROOT} STREQUAL "")
  set(OMPT_ROOT $ENV{OMPT_ROOT})
endif()

pkg_check_modules(PC_OMPT QUIET OMPT)
set(OMPT_DEFINITIONS ${PC_OMPT_CFLAGS_OTHER})

find_path(OMPT_INCLUDE_DIR omp-tools.h
          HINTS ${PC_OMPT_INCLUDEDIR} ${PC_OMPT_INCLUDE_DIRS} ${OMPT_ROOT}/include)

find_library(OMPT_LIBRARY NAMES omp iomp5 gomp
             HINTS ${PC_OMPT_LIBDIR} ${PC_OMPT_LIBRARY_DIRS} ${OMPT_ROOT}/lib
			 ${OMPT_ROOT}/lib/* NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OMPT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OMPT  DEFAULT_MSG
                                  OMPT_LIBRARY OMPT_INCLUDE_DIR)

mark_as_advanced(OMPT_INCLUDE_DIR OMPT_LIBRARY)

set(APEX_OMPT_EXTRA_CONFIG "")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
    if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        message(ERROR " GCC version must be 8 or less to enable target offload.!")
        set(APEX_OMPT_EXTRA_CONFIG "-DOPENMP_ENABLE_LIBOMPTARGET=FALSE")
    endif()
endif()

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if(APEX_BUILD_OMPT OR (NOT OMPT_FOUND))
  set(CACHE OMPT_ROOT ${CMAKE_INSTALL_PREFIX} STRING "OMPT Root directory")
  message("Attention: Downloading and Building OMPT as external project!")
  message(INFO " A working internet connection is required!")
  include(ExternalProject)
  ExternalProject_Add(project_ompt
      #GIT_REPOSITORY https://github.com/OpenMPToolsInterface/LLVM-openmp.git
      GIT_REPOSITORY https://github.com/llvm-mirror/openmp.git
	GIT_TAG master
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/LLVM-ompt-5.0
    CONFIGURE_COMMAND cmake -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=Release ${APEX_OMPT_EXTRA_CONFIG} ../project_ompt
    BUILD_COMMAND make libomp-needed-headers all
    INSTALL_COMMAND make install
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  set(OMPT_ROOT ${CMAKE_INSTALL_PREFIX})
  #ExternalProject_Get_Property(project_ompt install_dir)
  add_library(omp SHARED IMPORTED)
  set_property(TARGET omp PROPERTY IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libomp.so)
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
endif(${APEX_HAVE_OMPT_NATIVE})

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(OMPT_FOUND)
  set(OMPT_LIBRARIES ${OMPT_LIBRARY} )
  set(OMPT_INCLUDE_DIRS ${OMPT_INCLUDE_DIR} )
  set(OMPT_DIR ${OMPT_ROOT})
  add_definitions(-DAPEX_HAVE_OMPT)
endif()

