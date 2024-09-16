# The name of our project is "APEX". CMakeLists files in this project can
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and
# to the root binary directory of the project as ${APEX_BINARY_DIR}.

cmake_minimum_required (VERSION 3.20.1 FATAL_ERROR)
# set the project name
project (APEX CXX C)

# The version number.
set (APEX_DESCRIPTION "Autonomic Performance Environment for eXascale" CACHE STRING "APEX project description")
set (APEX_VERSION_MAJOR 2 CACHE STRING "APEX Major Version")
set (APEX_VERSION_MINOR 6 CACHE STRING "APEX Minor Version")
set (APEX_VERSION_PATCH 5 CACHE STRING "APEX Patch Version")
set (APEX_HOMEPAGE_URL "http://github.com/UO-OACISS/apex" CACHE STRING "APEX homepage URL")

cmake_policy(VERSION 2.8.12)
if (${CMAKE_MAJOR_VERSION} GREATER 2)
  cmake_policy(SET CMP0042 NEW)
    if (${CMAKE_MINOR_VERSION} GREATER 11)
        cmake_policy(SET CMP0074 NEW)
    else()
        if (${CMAKE_MINOR_VERSION} GREATER 8)
            cmake_policy(SET CMP0054 OLD)
        endif()
    endif()
endif()

set (CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT OFF)
set (CMAKE_CXX_STANDARD_COMPUTED_DEFAULT 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
include(APEX_DefaultOptions)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
     message(FATAL_ERROR "GCC version must be at least 4.9!")
  endif()
endif()

# if the user didn't specify, set the installation directory.
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE STRING "APEX Installation directory" FORCE)
endif()

# check if we are building for Kitten / LXK, and force some settings.

if((DEFINED APEX_LXK_KITTEN) AND (APEX_LXK_KITTEN))
    SET(BUILD_STATIC_EXECUTABLES TRUE CACHE BOOL "Build Static Libraries and Executables" FORCE)
    SET(READ_PROC_FILESYSTEM FALSE CACHE BOOL "Periodically read the /proc filesystem status" FORCE)
endif()

################################################################################
# Set the RPATH to make sure executables always work without "make install"
################################################################################
if (NOT BUILD_STATIC_EXECUTABLES)
    # use, i.e. don't skip the full RPATH for the build tree
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

    # when building, don't use the install RPATH already
    # (but later on when installing)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # the RPATH to be used when installing, but only if it's not a system directory
    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    IF("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF("${isSystemDir}" STREQUAL "-1")
endif()

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  if(NOT DEFINED APEX_BUILD_TESTS)
    set(APEX_BUILD_TESTS TRUE)
  endif(NOT DEFINED APEX_BUILD_TESTS)
  if(NOT DEFINED APEX_BUILD_EXAMPLES)
    set(APEX_BUILD_EXAMPLES TRUE)
  endif(NOT DEFINED APEX_BUILD_EXAMPLES)
  add_definitions(-DDEBUG)
  add_definitions(-DCMAKE_BUILD_TYPE=3)
  set(APEX_ERROR_HANDLING TRUE)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  if(NOT DEFINED APEX_BUILD_TESTS)
    set(APEX_BUILD_TESTS TRUE)
  endif(NOT DEFINED APEX_BUILD_TESTS)
  if(NOT DEFINED APEX_BUILD_EXAMPLES)
    set(APEX_BUILD_EXAMPLES TRUE)
  endif(NOT DEFINED APEX_BUILD_EXAMPLES)
  add_definitions(-DNDEBUG)
  add_definitions(-DCMAKE_BUILD_TYPE=2)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_definitions(-DNDEBUG)
  # they will still be configured, but excluded from "all"
  if(NOT DEFINED APEX_BUILD_TESTS)
    set(APEX_BUILD_TESTS FALSE)
  endif(NOT DEFINED APEX_BUILD_TESTS)
  if(NOT DEFINED APEX_BUILD_EXAMPLES)
    set(APEX_BUILD_EXAMPLES FALSE)
  endif(NOT DEFINED APEX_BUILD_EXAMPLES)
  add_definitions(-DCMAKE_BUILD_TYPE=1)
endif()

if(APEX_WITH_CUDA)
    # This is needed so that the nvcc compiler test passes, and we can use it to
    # compile the test cases.
    if(APEX_BUILD_TESTS)
        if ( (CMAKE_CXX_COMPILER_ID MATCHES GNU) AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64.*$") )
            if ( (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0) )
                set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -std=c++11")
            endif ()
        endif ()
	#        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
	#    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
	#    message(INFO " Building CUPTI support, but GCC is too new for CUDA tests")
	#elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
            message(INFO " Building CUPTI support, but Clang < 11 can't build CUDA tests")
        else()
            # OK to enable CUDA tests
            enable_language(CUDA)
            set(APEX_CUDA_TESTS TRUE)
        endif()
    endif()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Mint128")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Mint128")
    endif()
    find_package(CUDAToolkit REQUIRED QUIET COMPONENTS CUPTI)
    #message(INFO " getting path to: ${CUDA_CUDART}")
    #get_filename_component(CUDA_LIBRARY_DIR ${CUDA_CUDART} DIRECTORY)
    #set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${CUDA_LIBRARY_DIR})
    #message(INFO " getting path to: ${CUDA_cupti_LIBRARY}")
    #get_filename_component(CUPTI_LIBRARY_DIR ${CUDA_cupti_LIBRARY} DIRECTORY)
    #set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${CUPTI_LIBRARY_DIR})
endif(APEX_WITH_CUDA)

if(APEX_WITH_HIP)
    # First, check if the user has set ROCM_ROOT.
    if(NOT DEFINED ROCM_ROOT)
        find_program(HIPCC_COMPILER hipcc)
        if (HIPCC_COMPILER)
            get_filename_component(hipcc_bin_dir "${HIPCC_COMPILER}" DIRECTORY)
            get_filename_component(hipcc_dir "${hipcc_bin_dir}" DIRECTORY)
            set(ROCM_ROOT "${hipcc_dir}" CACHE PATH "" FORCE)
            unset(hipcc_dir)
            unset(hipcc_bin_dir)
        endif (HIPCC_COMPILER)
    endif(NOT DEFINED ROCM_ROOT)

    # This is needed so that the clang compiler test passes, and we can use it to
    # compile the test cases.
    if(APEX_BUILD_TESTS)
        # Don't do this without hipcc!  It will try to compile all of APEX with
        # hipcc / clang++ and that for some reason is a hot mess...
        if(CMAKE_CXX_COMPILER_ID STREQUAL "ROCMClang"
             OR CMAKE_CXX_COMPILER MATCHES ".*hipcc")
            # OK to enable HIP tests
            find_package(hip REQUIRED)
            set(APEX_HIP_TESTS TRUE)
#        else()
#            enable_language(HIP)
#            set(APEX_HIP_TESTS TRUE)
        endif()
    endif()
endif(APEX_WITH_HIP)

if(APEX_WITH_LEVEL0)
    find_package(LEVEL0 REQUIRED)
    if (LEVEL0_FOUND)
        include_directories(${LEVEL0_INCLUDE_DIRS})
        add_definitions(-DAPEX_WITH_LEVEL0)
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${LEVEL0_LIBRARY_DIR})
        endif()
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter -Rno-debug-disables-optimization")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Rno-debug-disables-optimization")
    endif()
endif(APEX_WITH_LEVEL0)

# set(APEX_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)

# Always force CMAKE_CONFIGURATION_TYPES to be the same as CMAKE_BUILD_TYPE
# (at least for now, until we figure out how to use multiple build types in
# the same project).
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE INTERNAL
"Configuration types" FORCE)

#if(APPLE)
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -force_flat_namespace -undefined dynamic_lookup")
#  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
#endif(APPLE)

# If TAU is used, don't allow throttling - it can potentially lead to
# overlapping timer errors in TAU.
if((DEFINED APEX_THROTTLE) AND (APEX_THROTTLE))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_THROTTLE")
endif()

if((DEFINED APEX_LXK_KITTEN) AND (APEX_LXK_KITTEN))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_LXK_KITTEN")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPEX_LXK_KITTEN")
endif()

if((DEFINED APEX_DEBUG) AND (APEX_DEBUG))
  add_definitions(-DAPEX_DEBUG)
endif((DEFINED APEX_DEBUG) AND (APEX_DEBUG))

# include additional macro definitions

include(APEX_Utils)
include(APEX_Distclean)

apex_force_out_of_tree_build("This project requires an out-of-source-tree
build. See README.rst. Clean your CMake cache and CMakeFiles if this message
persists.")

################################################################################
# Get the GIT version of the code
################################################################################

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the current working tag
execute_process(
  COMMAND git describe --abbrev=0 --tags
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

#add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
#add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

################################################################################
# Hostname detection code
################################################################################

site_name(HOSTNAME)
string(LENGTH ${HOSTNAME} HOSTNAME_LENGTH)
if(${HOSTNAME_LENGTH} GREATER 5)
    string(SUBSTRING ${HOSTNAME} 0 6 HOST_BASENAME)
else()
    set (HOST_BASENAME ${HOSTNAME})
endif()

################################################################################
# Compiler detection code
################################################################################

# SET SANITIZE OPTIONS, IF DESIRED

# defaults
set(APEX_SANITIZE_OPTIONS "")

# memory, other
if (DEFINED APEX_SANITIZE AND APEX_SANITIZE)
  #set(APEX_SANITIZE_OPTIONS "-fsanitize=address -fsanitize=undefined -fsanitize-recover=address ")
  set(APEX_SANITIZE_OPTIONS "-fsanitize=address -fsanitize-recover=address ")
endif (DEFINED APEX_SANITIZE AND APEX_SANITIZE)

# race conditions
if (DEFINED APEX_SANITIZE_THREAD AND APEX_SANITIZE_THREAD)
  set(APEX_SANITIZE_OPTIONS "-fsanitize=thread -fsanitize-recover=thread ")
endif (DEFINED APEX_SANITIZE_THREAD AND APEX_SANITIZE_THREAD)

# set debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${APEX_SANITIZE_OPTIONS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${APEX_SANITIZE_OPTIONS}")

if (DEFINED APEX_ERROR_HANDLING AND APEX_ERROR_HANDLING)
    add_definitions(-DAPEX_ERROR_HANDLING)
endif (DEFINED APEX_ERROR_HANDLING AND APEX_ERROR_HANDLING)

# ---------------------------------------------------
# Set pedantic error flags if available
# ---------------------------------------------------
include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            message(INFO " ${flag} supported by compiler, enabling it")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

# example usage
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
# Great, the GCC 9 compiler has a pedantic warning error.
if (APEX_USE_PEDANTIC)
    enable_cxx_compiler_flag_if_supported("-Werror")
    enable_cxx_compiler_flag_if_supported("-Wpedantic")
    enable_cxx_compiler_flag_if_supported("-pedantic")
    if(APEX_WITH_HIP)
        enable_cxx_compiler_flag_if_supported("-Wno-gnu-anonymous-struct")
        enable_cxx_compiler_flag_if_supported("-Wno-nested-anon-types")
        enable_cxx_compiler_flag_if_supported("-Wno-missing-field-initializers")
        enable_cxx_compiler_flag_if_supported("-Wno-vla-extension")
        enable_cxx_compiler_flag_if_supported("-Wno-unused-parameter")
    endif(APEX_WITH_HIP)
    if(APEX_WITH_CUDA)
        enable_cxx_compiler_flag_if_supported("--diag_suppress bad_pp_directive_keyword")
        enable_cxx_compiler_flag_if_supported("--diag_suppress unrecognized_attribute")
        enable_cxx_compiler_flag_if_supported("--diag_suppress set_but_not_used")
        enable_cxx_compiler_flag_if_supported("--diag_suppress omitted_exception_specification")
    endif(APEX_WITH_CUDA)
endif (APEX_USE_PEDANTIC)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    set(APEX_STDCXX_LIB "" CACHE STRING "C++ library for linking")
    message("IntelLLVM compiler detected, no stdc++ library needed")
else()
    set(APEX_STDCXX_LIB stdc++ CACHE STRING "C++ library for linking")
endif()

# PGI and Intel don't like the concurrentqueue code.
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # ROCM Clang++ has an unused library in the link.
        if (NOT ${APEX_WITH_HIP})
            #enable_cxx_compiler_flag_if_supported("-Werror")
        endif (NOT ${APEX_WITH_HIP})
    endif (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")

# ---------------------------------------------------

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_definitions(-std=c++11)
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
#if(COMPILER_SUPPORTS_CXX17)
#    set(CMAKE_CXX_SUPPORT_FLAG "-std=c++17" CACHE STRING "CXX Support Flag" FORCE)
#else()
#    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
#    if(COMPILER_SUPPORTS_CXX14)
#        set(CMAKE_CXX_SUPPORT_FLAG "-std=c++14" CACHE STRING "CXX Support Flag" FORCE)
#    else()
#        CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
#        if(COMPILER_SUPPORTS_CXX1Y)
#            set(CMAKE_CXX_SUPPORT_FLAG "-std=c++1y" CACHE STRING "CXX Support Flag" FORCE)
#        else()
#            CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#            if(COMPILER_SUPPORTS_CXX11)
#                set(CMAKE_CXX_SUPPORT_FLAG "-std=c++11" CACHE STRING "CXX Support Flag" FORCE)
#            else()
#                CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#                if(COMPILER_SUPPORTS_CXX0X)
#                    set(CMAKE_CXX_SUPPORT_FLAG "-std=c++0x" CACHE STRING "CXX Support Flag" FORCE)
#                else()
#                    message(FATAL_ERROR " Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
#                endif()
#            endif()
#        endif()
#    endif()
#endif()

# By the way, GCC lies.  It accepts the flag, but doesn't have the support.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message("Forcing C++ standard to c++11 for GCC : ${CMAKE_CXX_COMPILER_VERSION}")
        set(CMAKE_CXX_SUPPORT_FLAG "-std=c++11" CACHE STRING "CXX Support Flag" FORCE)
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.1)
        message("Forcing C++ standard to c++14 for GCC : ${CMAKE_CXX_COMPILER_VERSION}")
        set(CMAKE_CXX_SUPPORT_FLAG "-std=c++14" CACHE STRING "CXX Support Flag" FORCE)
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_SUPPORT_FLAG}")

# Check if architecture is x86 or not
message("System architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)")
  set(APEX_ARCH_X86 TRUE)
else()
  set(APEX_ARCH_X86 FALSE)
endif()

if (BUILD_STATIC_EXECUTABLES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(CMAKE_POSITION_INDEPENDENT_CODE FALSE)
    set(POSITION_INDEPENDENT_CODE FALSE)
    set(BUILD_SHARED_LIBS FALSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_STATIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPEX_STATIC")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
    set(APEX_WITH_JEMALLOC FALSE)
    unset(JEMALLOC_ROOT)
    set(APEX_WITH_TCMALLOC FALSE)
    unset(GPERFTOOLS_ROOT)
else()
    # use PIC for shared objects
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
    set(POSITION_INDEPENDENT_CODE TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

################################################################################
# Boost configuration
################################################################################

# BOOST IS NOW ONLY REQUIRED FOR INTEL XEON MIC/PHI SUPPORT!
if(APEX_INTEL_MIC)
    # reconfigure Boost library information if configuration type has been
    # changed since last configure
    if(CMAKE_BUILD_TYPE AND NOT (CMAKE_BUILD_TYPE STREQUAL CMAKE_CONFIGURATION_TYPES))
        set(BOOST_VERSION_SEARCHED OFF CACHE INTERNAL "Found Boost version" FORCE)
        set(BOOST_SEARCHED OFF CACHE INTERNAL "Found Boost libraries" FORCE)
    endif()

    set(Boost_USE_MULTITHREADED ON)
    if (BUILD_STATIC_EXECUTABLES)
        set(Boost_USE_STATIC_LIBS ON)
        set(Boost_USE_STATIC_RUNTIME ON)
    else()
        set(Boost_USE_STATIC_LIBS OFF)
        set(Boost_USE_STATIC_RUNTIME OFF)
    endif()

    if(BUILD_BOOST)
        include(ExternalProjectBoost)
        build_boost_project()
        include_directories(${Boost_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${Boost_LIBRARIES})
        message(WARNING " Downloading and building Boost!")
    else()
        find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread regex)
        if(Boost_FOUND)
            include_directories(${Boost_INCLUDE_DIRS})
            set(LIBS ${LIBS} ${Boost_LIBRARIES})
        else()
            message(FATAL_ERROR " Boost not found. To download and build Boost, add the cmake argument -DBUILD_BOOST=TRUE")
        endif()
        add_custom_target(project_boost)
    endif()
endif(APEX_INTEL_MIC)

################################################################################
# MPI configuration
################################################################################

if(APEX_WITH_MPI)
  enable_language(Fortran)
  find_package(MPI REQUIRED)
endif()

if(MPI_CXX_FOUND)
    add_definitions(-DAPEX_WITH_MPI)
    # Prevent errors from Cmake by stripping whitespace
    string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
    string(STRIP "${MPI_CXX_LIBRARIES}" MPI_CXX_LIBRARIES)
    set(LIBS ${LIBS} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

################################################################################
# RCR configuration
################################################################################

# RCR is NOT required.
if (DEFINED RCR_ROOT OR DEFINED $ENV{RCR_ROOT})
    find_package(RCR)
endif()

if (RCR_FOUND)
    include_directories(${RCR_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${RCR_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${RCR_LIBRARY_DIR})
    endif()
    add_definitions(-DAPEX_HAVE_RCR)
else()
    if(EXISTS "/sys/cray/pm_counters/power")
    #IF("$ENV{CRAY_CPU_TARGET}" STREQUAL "mic-knl")
        #add_definitions(-fPIC)
        set (APEX_HAVE_CRAY_POWER TRUE)
        add_definitions(-DAPEX_HAVE_CRAY_POWER)
        message(INFO " System has Cray energy monitoring support.")
    else()
      if(EXISTS "/sys/class/powercap/intel-rapl/intel-rapl:0")
        set (APEX_HAVE_POWERCAP_POWER TRUE)
        add_definitions(-DAPEX_HAVE_POWERCAP_POWER)
        message(INFO " System has Powercap energy monitoring support.")
      endif()
    endif()
endif()

################################################################################
# OMPT configuration
################################################################################

find_package(OpenMP)
find_package(OpenACC)
if(OpenACC_CXX_FOUND)
    find_package(OpenACCProfiling)
endif()

if((DEFINED APEX_WITH_OMPT) AND (NOT APEX_WITH_OMPT))
    # just in case, disable all ompt
    unset(OMPT_ROOT)
    unset(ENV{OMPT_ROOT})
    unset(APEX_BUILD_OMPT)
endif()

#if ((NOT DEFINED APEX_WITH_OMPT) OR (APEX_WITH_OMPT) OR (APEX_BUILD_OMPT))
if ((NOT DEFINED APEX_WITH_OMPT) OR (APEX_WITH_OMPT))
    if (OPENMP_FOUND)
        #if ((DEFINED OMPT_ROOT) OR (APEX_WITH_OMPT) OR (APEX_BUILD_OMPT))
        if ((DEFINED OMPT_ROOT) OR (APEX_WITH_OMPT))
            find_package(OMPT REQUIRED)
            if (OMPT_FOUND)
#                include_directories(${OMPT_INCLUDE_DIRS})
#                set(LIBS ${LIBS} ${OMPT_LIBRARIES})
#                if (NOT BUILD_STATIC_EXECUTABLES)
#                    set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OMPT_LIBRARY_DIR})
#                endif()
            else()
                message(FATAL_ERROR "Your compiler does not have OpenMP Tools support, you can't enable APEX_USE_OMPT=TRUE.\nPlease disable that option or use a newer/better compiler.\n")
                #add_custom_target(project_ompt)
            endif()
        #else()
            #add_custom_target(project_ompt)
        endif()
    endif()
else()
    #add_custom_target(project_ompt)
endif ((NOT DEFINED APEX_WITH_OMPT) OR (APEX_WITH_OMPT))
#endif ((NOT DEFINED APEX_WITH_OMPT) OR (APEX_WITH_OMPT) OR (APEX_BUILD_OMPT))

################################################################################
# PROC configuration
################################################################################

if ((NOT DEFINED READ_PROC_FILESYSTEM) OR (READ_PROC_FILESYSTEM))
    if(EXISTS "/proc/stat")
        set (APEX_HAVE_PROC TRUE)
        add_definitions(-DAPEX_HAVE_PROC)
    endif()
endif()

################################################################################
# JE/TCMalloc configuration
################################################################################

if((DEFINED JEMALLOC_ROOT) OR (APEX_WITH_JEMALLOC))
    find_package(JEmalloc)
    if (JEmalloc_FOUND)
        include_directories(${JEmalloc_INCLUDE_DIRS})
        add_definitions(-DAPEX_WITH_JEMALLOC)
        set(LIBS ${LIBS} ${JEmalloc_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${JEmalloc_LIBRARY_DIR})
        endif()
    endif()
endif()

if((DEFINED GPERFTOOLS_ROOT) OR (APEX_WITH_TCMALLOC))
    find_package(Tcmalloc REQUIRED)
    if (Tcmalloc_FOUND)
        add_definitions(-DAPEX_HAVE_TCMALLOC)
        add_definitions(-DAPEX_WITH_TCMALLOC)
        include_directories(${Tcmalloc_INCLUDE_DIR})
        set(LIBS ${LIBS} ${Tcmalloc_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${Tcmalloc_LIBRARY_DIR})
        endif()
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
        endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    else()
        add_custom_target(project_gperftools)
    endif()
else()
    add_custom_target(project_gperftools)
endif()

################################################################################
# LM Sensors configuration
################################################################################

if((DEFINED LM_SENSORS_ROOT) OR (APEX_WITH_LM_SENSORS))
    find_package(LM_SENSORS)
    if (LM_SENSORS_FOUND)
        include_directories(${LM_SENSORS_INCLUDE_DIRS})
        add_definitions(-DAPEX_WITH_LM_SENSORS)
        set(LIBS ${LIBS} ${LM_SENSORS_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${LM_SENSORS_LIBRARY_DIR})
        endif()
    endif()
endif()

################################################################################
# ActiveHarmony configuration
################################################################################

find_package(ACTIVEHARMONY)
if (ACTIVEHARMONY_FOUND)
    include_directories(${ACTIVEHARMONY_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ACTIVEHARMONY_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ACTIVEHARMONY_LIBRARY_DIR})
    endif()
endif()

# Just in case, to prevent concurrent builds
add_dependencies (project_activeharmony project_gperftools)

################################################################################
# PAPI configuration
################################################################################

if((DEFINED PAPI_ROOT) OR (APEX_WITH_PAPI))
    find_package(PAPI REQUIRED)
    if (PAPI_FOUND)
        message(INFO " Using PAPI include: ${PAPI_INCLUDE_DIRS}")
        message(INFO " Using PAPI libraries: ${PAPI_LIBRARIES}")
        include_directories(${PAPI_INCLUDE_DIRS})
        add_definitions(-DAPEX_WITH_PAPI)
        add_definitions(-DAPEX_HAVE_PAPI)
        set(LIBS ${LIBS} ${PAPI_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${PAPI_LIBRARY_DIR})
        endif()
    endif()
endif()

################################################################################
# Kokkos configuration
################################################################################

if(APEX_WITH_KOKKOS)
    message(INFO " Checking for Kokkos installation in $Kokkos_ROOT...")
    find_package(Kokkos)
    if (Kokkos_FOUND)
        message(INFO " Using Kokkos include: ${Kokkos_INCLUDE_DIRS}/impl")
        include_directories(${Kokkos_INCLUDE_DIRS})
        include_directories(${Kokkos_INCLUDE_DIRS}/impl)
        # Can only build Kokkos test with an installed Kokkos.
    else()
        message(INFO " Kokkos not found, cloning submodule to get required headers.")
        include(cmake/Modules/AddGitSubmodule.cmake)
        add_git_submodule(kokkos FALSE)
        include_directories(${PROJECT_SOURCE_DIR}/kokkos/core/src/impl)
        include_directories(${Kokkos_INCLUDE_DIRS_RET})
        #include_directories(${Kokkos_INCLUDE_DIRS_RET}/impl)
        if(APEX_BUILD_TESTS)
            # Just for testing
            SET(Kokkos_LIBRARY kokkoscore)
            set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
            set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
            set(Kokkos_ARCH_NATIVE ON CACHE BOOL "" FORCE)
            set(Kokkos_ENABLE_TUNING ON CACHE BOOL "" FORCE)
            add_subdirectory(kokkos)
        endif(APEX_BUILD_TESTS)
    endif()
    add_definitions(-DAPEX_WITH_KOKKOS)
endif()

################################################################################
# OTF2 configuration
################################################################################

find_package(OTF2)
if (OTF2_FOUND)
    include_directories(${OTF2_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OTF2_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OTF2_LIBRARY_DIR})
    endif()
    if(NOT APPLE)
        find_library(RTLIB rt)
        set(LIBS ${LIBS} ${RTLIB})
    endif(NOT APPLE)
endif()

# Just in case, to prevent concurrent builds
add_dependencies (project_otf2 project_activeharmony)

################################################################################
# ZLIB configuration
################################################################################

find_package(ZLIB)
if(ZLIB_FOUND)
    set(LIBS ${LIBS} ${ZLIB_LIBRARIES})
    add_definitions(-DAPEX_HAVE_ZLIB)
    message(INFO " Using ZLIB include: ${ZLIB_INCLUDE_DIRS}")
    include_directories(${ZLIB_INCLUDE_DIRS})
endif(ZLIB_FOUND)

################################################################################
# CUDA configuration
################################################################################

if (APEX_WITH_CUDA)
    find_package(CUPTI REQUIRED)
    find_package(NVML REQUIRED)
    add_definitions(-DAPEX_WITH_CUDA)
    message(INFO " Using CUPTI include: ${CUPTI_INCLUDE_DIRS}")
    include_directories(${CUPTI_INCLUDE_DIRS})
    message(INFO " Using NVML include: ${NVML_INCLUDE_DIRS}")
    include_directories(${NVML_INCLUDE_DIRS})
endif (APEX_WITH_CUDA)

################################################################################
# HIP configuration
################################################################################

if (APEX_WITH_HIP)
    find_package(ROCTRACER REQUIRED)
    # As of 5.7.0, librocprofiler64.so is not found.
    find_package(ROCPROFILER)
    if(ROCPROFILER_FOUND)
        message(INFO " ROCPROFILER found!")
        add_definitions(-DAPEX_WITH_ROCPROFILER)
    else(ROCPROFILER_FOUND)
        message(INFO " ROCPROFILER not found!")
    endif(ROCPROFILER_FOUND)
    find_package(ROCTX REQUIRED)
    find_package(RSMI REQUIRED)
    add_definitions(-DAPEX_WITH_HIP)
    # add_definitions(-DAPEX_WITH_HSA)
    # add_definitions(-DAPEX_WITH_KFD)
    message(INFO " Using ROCTRACER include: ${ROCTRACER_INCLUDE_DIRS}")
    include_directories(${ROCTRACER_INCLUDE_DIRS})
    message(INFO " Using ROCPROFILER include: ${ROCPROFILER_INCLUDE_DIRS}")
    include_directories(${ROCPROFILER_INCLUDE_DIRS})
    message(INFO " Using ROCTX include: ${ROCTX_INCLUDE_DIRS}")
    include_directories(${ROCTX_INCLUDE_DIRS})
    message(INFO " Using ROCm SMI include: ${RSMI_INCLUDE_DIRS}")
    include_directories(${RSMI_INCLUDE_DIRS})
    # Annoyingly, the librocprofiler64.so library doesn't retain an RPATH
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ROCPROFILER_LIBRARY_DIR})

    find_package(ROCTRACER_KFD) # removed in ROCM 4.5.0
    if (ROCTRACER_KFD_FOUND)
        set(LIBS ${LIBS} ${ROCTRACER_KFD_LIBRARIES})
        include_directories(${ROCTRACER_KFD_INCLUDE_DIRS})
    endif (ROCTRACER_KFD_FOUND)

endif (APEX_WITH_HIP)

################################################################################
# RAJA configuration
################################################################################

if (APEX_WITH_RAJA)
    find_package(RAJA REQUIRED)
    if(RAJA_FOUND)
        set(LIBS ${LIBS} ${RAJA_LIBRARIES})
        add_definitions(-DAPEX_WITH_RAJA)
        message(INFO " Using RAJA include: ${RAJA_DIR}/include")
        include_directories(${RAJA_DIR}/include)
    else()
        message(ERROR " RAJA not found in ${RAJA_DIR}")
    endif()
endif (APEX_WITH_RAJA)

#################################################################################
# StarPU configuration
################################################################################

if (APEX_WITH_STARPU)
  find_package(HWLOC REQUIRED)
  if(HWLOC_FOUND)
    set(LIBS ${LIBS} ${HWLOC_LIBRARIES})
    add_definitions(-DAPEX_WITH_HWLOC)
    message(INFO " Using HWLOC include: ${HWLOC_INCLUDE_DIRS}")
    include_directories(${HWLOC_INCLUDE_DIRS})
    link_directories(${HWLOC_LIBRARY_DIRS})
  else()
    if (DEFINED HWLOC_ROOT)
        message(ERROR " HWLOC not found in ${HWLOC_ROOT}")
    else (DEFINED HWLOC_ROOT)
        message(ERROR " HWLOC not found! Please include the cmake option -DHWLOC_ROOT=/path/to/hwloc")
    endif (DEFINED HWLOC_ROOT)
  endif()
  find_package(STARPU REQUIRED)
  if(STARPU_FOUND)
    set(LIBS ${LIBS} ${STARPU_LIBRARIES})
    add_definitions(-DAPEX_WITH_STARPU)
    message(INFO " Using StarPU include: ${STARPU_INCLUDE_DIR}")
    include_directories(${STARPU_INCLUDE_DIR})
  else()
    if (DEFINED STARPU_ROOT)
        message(ERROR " StarPU not found in ${STARPU_ROOT}")
    else (DEFINED STARPU_ROOT)
        message(ERROR " StarPU not found! Please include the cmake option -DSTARPU_ROOT=/path/to/starpu")
    endif (DEFINED STARPU_ROOT)
  endif()
endif (APEX_WITH_STARPU)

###############################################################################
# PhiProf configuration
################################################################################

if (APEX_WITH_PHIPROF)
   find_package(PHIPROF REQUIRED)
   if (PHIPROF_FOUND)
      add_definitions(-DAPEX_WITH_PHIPROF)
      message(INFO " Using PhiProf include: ${PHIPROF_INCLUDE_DIR}")
      include_directories(${PHIPROF_INCLUDE_DIR})
      # We need MPI
      if (NOT MPI_CXX_FOUND)
         message( FATAL_ERROR "MPI is required with PhiProf" )
      endif (NOT MPI_CXX_FOUND)
   endif (PHIPROF_FOUND)
endif (APEX_WITH_PHIPROF)

###############################################################################
# MSR configuration
################################################################################

if((DEFINED MSR_ROOT) OR (USE_MSR))
    find_package(MSR)
    if(NOT MSR_FOUND)
        message(FATAL_ERROR " Requested USE_MSR but could not find MSR; set MSR_ROOT")
    endif()
    message(INFO " Using msr: ${MSR_LIBRARY_DIR} ${MSR_LIBRARIES}")
    include_directories(${MSR_INCLUDE_DIR})
    set(LIBS ${LIBS} ${MSR_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${MSR_LIBRARY_DIR})
    endif()
endif()

################################################################################
# Binutils configuration
################################################################################

if((DEFINED BFD_ROOT) OR (APEX_WITH_BFD) OR (APEX_BUILD_BFD))
    find_package(BFD)
    if (BFD_FOUND)
        include_directories(${BFD_INCLUDE_DIRS})
        add_definitions(-DAPEX_WITH_BFD)
        set(LIBS ${LIBS} ${BFD_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${BFD_LIBRARY_DIR})
        endif()
        message(INFO " Using binutils: ${BFD_LIBRARY_DIR} ${BFD_LIBRARIES}")
    endif()

    find_package(DEMANGLE)
    if (DEMANGLE_FOUND)
        include_directories(${DEMANGLE_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${DEMANGLE_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${DEMANGLE_LIBRARY_DIR})
        endif()
        message(INFO " Using demangle: ${DEMANGLE_LIBRARY_DIR} ${DEMANGLE_LIBRARIES}")
    else()
      unset(DEMANGLE_LIBRARY)
      unset(DEMANGLE_LIBRARIES)
      unset(DEMANGLE_DIR)
    endif()
else()
    add_custom_target(project_binutils)
endif()
# Just in case, to prevent concurrent builds
add_dependencies (project_binutils project_otf2)

if(BFD_FOUND OR OTF2_FOUND)
    if(NOT APEX_INTEL_MIC)
        find_package(ZLIB)
        if (ZLIB_FOUND)
            set(LIBS ${LIBS} ${ZLIB_LIBRARIES})
            if (NOT BUILD_STATIC_EXECUTABLES)
                set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ZLIB_LIBRARY_DIR})
            endif()
            message(INFO " Using zlib: ${ZLIB_LIBRARY_DIR} ${ZLIB_LIBRARIES}")
        endif()
    endif(NOT APEX_INTEL_MIC)
endif(BFD_FOUND OR OTF2_FOUND)

include(GitExternal)

### Set up concurrentqueue stuff
git_external(concurrentqueue
    https://github.com/khuck/concurrentqueue.git
    master
    VERBOSE NO_UPDATE)

find_file(
    CONCURRENTQUEUE_HEADER
    NAMES concurrentqueue.h
    PATHS ${PROJECT_SOURCE_DIR}/concurrentqueue)

if(CONCURRENTQUEUE_HEADER)
    message(INFO " Found concurrentqueue at ${PROJECT_SOURCE_DIR}/concurrentqueue")
    include_directories(${PROJECT_SOURCE_DIR})
else()
    message(FATAL_ERROR " concurrentqueue not found. This should have been checked out automatically. "
            "Try manually check out https://github.com/cameron314/concurrentqueue.git to ${PROJECT_SOURCE_DIR}")
endif()

### Set up perfstubs stuff
git_external(perfstubs
    https://github.com/khuck/perfstubs.git
    master
    VERBOSE)

find_file(
    PERFSTUBS_HEADER
    NAMES perfstubs_api/tool.h
    PATHS ${PROJECT_SOURCE_DIR}/perfstubs)

if(PERFSTUBS_HEADER)
    message(INFO " Found perfstubs at ${PROJECT_SOURCE_DIR}/perfstubs")
    include_directories(${PROJECT_SOURCE_DIR}/perfstubs)
else()
    message(FATAL_ERROR " perfstubs not found. This should have been checked out automatically. "
            "Try manually check out https://github.com/khuck/perfstubs.git to ${PROJECT_SOURCE_DIR}")
endif()

### Set up taskStubs stuff
git_external(taskStubs
    https://github.com/khuck/taskStubs.git
    main
    VERBOSE)

find_file(
    TASKSTUBS_HEADER
    NAMES timer_plugin/tool_api.h
    PATHS ${PROJECT_SOURCE_DIR}/taskStubs)

if(TASKSTUBS_HEADER)
    message(INFO " Found taskStubs at ${PROJECT_SOURCE_DIR}/taskStubs")
    include_directories(${PROJECT_SOURCE_DIR}/taskStubs)
else()
    message(FATAL_ERROR " taskStubs not found. This should have been checked out automatically. "
            "Try manually check out https://github.com/khuck/taskStubs.git to ${PROJECT_SOURCE_DIR}")
endif()

if(APEX_WITH_PLUGINS)
    message(INFO " apex will be built with plugin support.")
    set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
    add_definitions("-DAPEX_WITH_PLUGINS")
endif()

git_external(rapidjson
    https://github.com/miloyip/rapidjson.git
    master
    VERBOSE)

find_path(
    RAPIDJSON_INCLUDE_DIR
    NAMES rapidjson
    PATHS ${PROJECT_SOURCE_DIR}/rapidjson/include)

if(RAPIDJSON_INCLUDE_DIR)
    message(INFO " Found rapidjson at ${RAPIDJSON_INCLUDE_DIR}")
    include_directories(${RAPIDJSON_INCLUDE_DIR})
else()
    message(FATAL_ERROR " rapidjson not found. This should have been checked out automatically. "
        "Try manually check out https://github.com/miloyip/rapidjson.git to ${PROJECT_SOURCE_DIR}")
endif()

################################################################################
# Standard Library configuration
################################################################################

# checking for pthread the correct way is broken for MIC.

if(APEX_INTEL_MIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
elseif(DEFINED WINDOWS)
    set(CMAKE_THREAD_PREFER_PTHREAD OFF)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
    # add_definitions(-D_MSC_VER=1)
else()
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package (Threads)
    if (Threads_FOUND)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
    else (Threads_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    endif (Threads_FOUND)
endif(APEX_INTEL_MIC)

if (RCR_FOUND)
    if(NOT APPLE)
        find_library(RTLIB rt)
        set(LIBS ${LIBS} ${RTLIB})
    endif(NOT APPLE)
    find_library(STDLIBCPP stdc++)
    set(LIBS ${LIBS} ${STDLIBCPP})
endif()

# apparently, we need to make sure libm is last.
find_library(MATHLIB m)
set(LIBS ${LIBS} ${MATHLIB})
#message(${LIBS})

if (BFD_FOUND)
    find_library(DYNAMICLIB dl)
    set(LIBS ${LIBS} ${DYNAMICLIB})
endif()

################################################################################

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/src/apex/apex_config.h.in"
  "${PROJECT_BINARY_DIR}/src/apex/apex_config.h"
)

if (NOT BUILD_STATIC_EXECUTABLES)
    set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# message(INFO " configuring pkgconfig with ${LIBS}")

# needed for apex.pc.in
SET(prefix ${CMAKE_INSTALL_PREFIX})
SET(exec_prefix "\${prefix}")
SET(libdir "\${exec_prefix}/lib")
SET(bindir "\${exec_prefix}/bin")
SET(includedir "\${prefix}/include")
foreach(_lib ${LIBS})
    string(REGEX MATCH "^-" is_short "${_lib}")
    if ("${is_short}" STREQUAL "-")
        list(APPEND MY_LIBS "${_lib}")
    else()
        get_filename_component(_lib_name ${_lib} NAME_WE)
        string(REGEX REPLACE "^lib" "" _name ${_lib_name})
        list(APPEND MY_LIBS "-l${_name}")
        get_filename_component(_lib_dir ${_lib} PATH)
        if (NOT "${_lib_dir}" STREQUAL "")
            string(REGEX MATCH "^/usr/lib" is_system "${_lib_dir}")
            if (NOT "${is_system}" STREQUAL "/usr/lib")
                list(APPEND MY_LINKFLAGS "-L${_lib_dir}")
            endif()
        endif()
    endif()
endforeach()
# message(INFO " configuring pkgconfig with ${MY_LIBS}")
if (NOT "${MY_LIBS}" STREQUAL "")
    list(REMOVE_DUPLICATES MY_LIBS )
endif()
if (NOT "${MY_LINKFLAGS}" STREQUAL "")
    list(REMOVE_DUPLICATES MY_LINKFLAGS )
endif()
string(REPLACE ";" " " TMP_LIBS "${MY_LIBS}")
string(REPLACE ";" " " TMP_LINKFLAGS "${MY_LINKFLAGS}")
if (NOT BUILD_STATIC_EXECUTABLES)
    # absorb the spaces first
    string(REGEX REPLACE "[ ]*-L" ",-rpath," TMP_RPATH "${TMP_LINKFLAGS}")
    if(TMP_RPATH)
      set(extralibs_rpath "-Wl${TMP_RPATH}")
    else()
      set(extralibs_rpath "")
    endif()
endif()
#SET(extralibs "${TMP_LINKFLAGS} ${TMP_LIBS} ${CMAKE_EXE_LINKER_FLAGS}")
SET(extralibs "${TMP_LINKFLAGS} ${TMP_LIBS}")
if (RCR_FOUND)
SET(extraincludes "-I${RCR_INCLUDE_DIR}")
endif (RCR_FOUND)
SET(requirements "")
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/pkgconfig/apex.pc.in
	${PROJECT_BINARY_DIR}/pkgconfig/apex.pc @ONLY)
INSTALL_FILES(/lib/pkgconfig FILES pkgconfig/apex.pc)

if (APEX_USE_STATIC_GLOBAL_CONSTRUCTOR OR APPLE)
    add_definitions(-DAPEX_USE_STATIC_GLOBAL_CONSTRUCTOR)
endif()

if (APEX_USE_WEAK_SYMBOLS)
    add_definitions(-DAPEX_USE_WEAK_SYMBOLS)
else()
    find_library(DYNAMICLIB dl)
    set(LIBS ${LIBS} ${DYNAMICLIB})
endif (APEX_USE_WEAK_SYMBOLS)

message(INFO " Getting processor count...")
include(ProcessorCount) # how many cores at our disposal?
if(PROCESSOR_COUNT)
    set(CTEST_BUILD_FLAGS "-j${PROCESSOR_COUNT}")
endif()

# turn on "make test" support
enable_testing()

# Recurse into the "Apex" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
if(APEX_WITH_PERFETTO)
    add_definitions("-DAPEX_WITH_PERFETTO")
    add_subdirectory (src/perfetto_sdk)
else()
    add_custom_target(perfetto_target)
endif()
add_subdirectory (src/apex)
#add_subdirectory (src/ITTNotify)
add_subdirectory (src/comm)
if (NOT BUILD_STATIC_EXECUTABLES)
    add_subdirectory (src/wrappers)
endif()
if(MPI_CXX_FOUND)
  add_subdirectory (src/comm/mpi)
endif()
# Needed to configure apex_exec
set(APEX_LIBRARY_NAME libapex)
add_subdirectory (src/scripts)

add_subdirectory (doc)

add_custom_target(tests)
add_custom_target(examples)
if(NOT MSVC)
    add_custom_command(TARGET tests POST_BUILD COMMAND ctest -R test --output-on-failure --timeout 100)
    if(APEX_HAVE_CRAY_POWER)
        # don't run the MPI examples on nersc head nodes - the admins get angry.
        add_custom_command(TARGET examples POST_BUILD COMMAND ctest -R Example --output-on-failure --timeout 100)
    endif()
endif()

add_subdirectory (src/utils)

SET(APEX_EXT_LIBRARIES APEX::apex APEX::apex_ompt APEX::apex_hip APEX::apex_cuda)
SET(APEX_SUB_LIBRARIES apex taudummy)
IF (APEX_WITH_OMPT)
  LIST(APPEND APEX_EXT_LIBRARIES APEX::apex_ompt)
  LIST(APPEND APEX_SUB_LIBRARIES apex_ompt)
ENDIF()
IF (APEX_WITH_CUDA)
  LIST(APPEND APEX_EXT_LIBRARIES APEX::apex_cuda)
  LIST(APPEND APEX_SUB_LIBRARIES apex_cuda)
ENDIF()
IF (APEX_WITH_HIP)
  LIST(APPEND APEX_EXT_LIBRARIES APEX::apex_hip)
  LIST(APPEND APEX_SUB_LIBRARIES apex_hip)
ENDIF()
IF (APEX_WITH_LEVEL0)
  LIST(APPEND APEX_EXT_LIBRARIES APEX::apex_level0)
  LIST(APPEND APEX_SUB_LIBRARIES apex_level0)
ENDIF()
SET(APEX_INT_LIBRARIES apex ${APEX_SUB_LIBRARIES})
SET_PROPERTY(GLOBAL PROPERTY APEX_INT_LIBRARIES ${APEX_INT_LIBRARIES})

SET(APEX_HEADER_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
SET(APEX_IS_SUBDIRECTORY FALSE)


if(APEX_BUILD_TESTS)
    message(INFO " Building unit tests.")
    if(OMPT_FOUND)
        add_subdirectory (src/openmp)
    endif(OMPT_FOUND)
    if(APEX_WITH_MPI)
        add_subdirectory (src/unit_tests/MPI)
    endif(APEX_WITH_MPI)
    add_subdirectory (src/unit_tests/C)
    add_subdirectory (src/unit_tests/C++)
    if (APEX_CUDA_TESTS)
        add_subdirectory (src/unit_tests/CUDA)
    endif (APEX_CUDA_TESTS)
    if (APEX_HIP_TESTS)
        add_subdirectory (src/unit_tests/HIP)
    endif (APEX_HIP_TESTS)
    if (APEX_WITH_KOKKOS)
        add_subdirectory (src/unit_tests/Kokkos)
    endif (APEX_WITH_KOKKOS)
else()
    #add_subdirectory (src/unit_tests/C EXCLUDE_FROM_ALL)
    #add_subdirectory (src/unit_tests/C++ EXCLUDE_FROM_ALL)
endif(APEX_BUILD_TESTS)

if(APEX_BUILD_EXAMPLES)
  message(INFO " Building examples.")
  add_subdirectory (src/examples)
else()
    #add_subdirectory (src/examples EXCLUDE_FROM_ALL)
endif(APEX_BUILD_EXAMPLES)

function(dump_cmake_variables)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        if (ARGV0)
            unset(MATCHED)
            string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
            if (NOT MATCHED)
                continue()
            endif()
        endif()
        message(STATUS "${_variableName} = ${${_variableName}}")
    endforeach()
endfunction()

function(dump_defines)
    get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
    foreach( d ${DirDefs} )
        message( STATUS "Found Define: " ${d} )
    endforeach()
endfunction()

message(STATUS "----------------------------------------------------------------------")
message(STATUS "APEX Variable Report:")
message(STATUS "----------------------------------------------------------------------")
dump_cmake_variables("^APEX")
message(STATUS "----------------------------------------------------------------------")
message(STATUS "APEX Dependency Report:")
message(STATUS "----------------------------------------------------------------------")
dump_cmake_variables("^BFD")
dump_cmake_variables("^CUDA")
dump_cmake_variables("^HIP")
dump_cmake_variables("^MPI")
dump_cmake_variables("^OMPT")
dump_cmake_variables("^OTF2")
dump_cmake_variables("^PAPI")
dump_cmake_variables("^STARPU")
dump_cmake_variables("^PHIPROF")
dump_cmake_variables("^HWLOC")
dump_defines()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Libraries: " ${LIBS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "Install Prefix:" ${CMAKE_INSTALL_PREFIX})
message(STATUS "----------------------------------------------------------------------")
