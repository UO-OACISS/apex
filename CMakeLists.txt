# The name of our project is "APEX". CMakeLists files in this project can
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and
# to the root binary directory of the project as ${APEX_BINARY_DIR}.

# set the project name
project (APEX CXX C)

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
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

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
     message(FATAL_ERROR "GCC version must be at least 4.9!")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

if(APEX_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED QUIET COMPONENTS CUPTI)
endif(APEX_USE_CUDA)

# if the user didn't specify, set the installation directory.
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE STRING "APEX Installation directory" FORCE)
endif()

# check if we are building for Kitten / LXK, and force some settings.

if((DEFINED APEX_LXK_KITTEN) AND (APEX_LXK_KITTEN))
    SET(BUILD_STATIC_EXECUTABLES TRUE CACHE BOOL "Build Static Libraries and Executables" FORCE)
    SET(READ_PROC_FILESYSTEM FALSE CACHE BOOL "Periodically read the /proc filesystem status" FORCE)
    SET(USE_CLOCK_TIMESTAMP TRUE CACHE BOOL "Disable assembly code timestamp call" FORCE)
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
  set(BUILD_EXAMPLES TRUE)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(BUILD_TESTS TRUE)
  set(BUILD_EXAMPLES TRUE)
  add_definitions(-DDEBUG)
  #add_definitions(-DAPEX_DEBUG)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  add_definitions(-DDEBUG)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_definitions(-DNDEBUG)
  # they will still be configured, but excluded from "all"
  if(NOT DEFINED BUILD_TESTS)
    set(BUILD_TESTS FALSE)
  endif(NOT DEFINED BUILD_TESTS)
  if(NOT DEFINED BUILD_EXAMPLES)
    set(BUILD_EXAMPLES FALSE)
  endif(NOT DEFINED BUILD_EXAMPLES)
endif()

# set(APEX_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)

# Always force CMAKE_CONFIGURATION_TYPES to be the same as CMAKE_BUILD_TYPE
# (at least for now, until we figure out how to use multiple build types in
# the same project).
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE INTERNAL
"Configuration types" FORCE)

if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
  set(USE_CLOCK_TIMESTAMP ON CACHE INTERNAL "Use gettimeofday not cycle counter" FORCE)
endif(APPLE)

# If TAU is used, don't allow throttling - it can potentially lead to
# overlapping timer errors in TAU.
if((DEFINED APEX_THROTTLE) AND (APEX_THROTTLE))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_THROTTLE")
endif()

if((DEFINED USE_CLOCK_TIMESTAMP) AND (USE_CLOCK_TIMESTAMP))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_USE_CLOCK_TIMESTAMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPEX_USE_CLOCK_TIMESTAMP")
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
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the current working tag
execute_process(
  COMMAND git describe --abbrev=0 --tags
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
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
  set(APEX_SANITIZE_OPTIONS "-fsanitize=address -fsanitize=undefined ")
endif (DEFINED APEX_SANITIZE AND APEX_SANITIZE)

# race conditions
if (DEFINED APEX_SANITIZE_THREAD AND APEX_SANITIZE_THREAD)
  set(APEX_SANITIZE_OPTIONS "-fsanitize=thread ")
endif (DEFINED APEX_SANITIZE_THREAD AND APEX_SANITIZE_THREAD)

# set debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${APEX_SANITIZE_OPTIONS} -DAPEX_ERROR_HANDLING")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${APEX_SANITIZE_OPTIONS} -DAPEX_ERROR_HANDLING")

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
enable_cxx_compiler_flag_if_supported("-Werror")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-Wpedantic")
enable_cxx_compiler_flag_if_supported("-pedantic")
# ---------------------------------------------------

# The version number.
set (APEX_VERSION_MAJOR 2)
set (APEX_VERSION_MINOR 1)

# add_definitions(-std=c++11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_SUPPORT_FLAG "-std=c++17" CACHE STRING "CXX Support Flag" FORCE)
else()
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_SUPPORT_FLAG "-std=c++14" CACHE STRING "CXX Support Flag" FORCE)
    else()
        CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
        if(COMPILER_SUPPORTS_CXX1Y)
            set(CMAKE_CXX_SUPPORT_FLAG "-std=c++1y" CACHE STRING "CXX Support Flag" FORCE)
        else()
            CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
            if(COMPILER_SUPPORTS_CXX11)
                set(CMAKE_CXX_SUPPORT_FLAG "-std=c++11" CACHE STRING "CXX Support Flag" FORCE)
            else()
                CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
                if(COMPILER_SUPPORTS_CXX0X)
                    set(CMAKE_CXX_SUPPORT_FLAG "-std=c++0x" CACHE STRING "CXX Support Flag" FORCE)
                else()
                    message(FATAL_ERROR " Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
                endif()
            endif()
        endif()
    endif()
endif()

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

# If not x86, don't use RDTSC
if(NOT APEX_ARCH_X86 OR CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
  add_definitions(-DAPEX_USE_CLOCK_TIMESTAMP=1)
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
    set(USE_JEMALLOC FALSE)
    unset(JEMALLOC_ROOT)
    set(USE_TCMALLOC FALSE)
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

if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(MPI_CXX_FOUND)
    add_definitions(-DAPEX_HAVE_MPI)
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
    #if(EXISTS "/sys/cray/pm_counter/power")
    IF(("${HOST_BASENAME}" STREQUAL "edison") OR ("$ENV{NERSC_HOST}" STREQUAL "edison") OR
       ("${HOST_BASENAME}" STREQUAL "cori") OR ("$ENV{NERSC_HOST}" STREQUAL "cori"))
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

if((DEFINED USE_OMPT) AND (NOT USE_OMPT))
    # just in case, disable all ompt
    unset(OMPT_ROOT)
    unset(ENV{OMPT_ROOT})
    unset(BUILD_OMPT)
endif()

if ((NOT DEFINED USE_OMPT) OR (USE_OMPT) OR (BUILD_OMPT))
    if (OPENMP_FOUND)
        if ((DEFINED OMPT_ROOT) OR (USE_OMPT) OR (BUILD_OMPT))
            find_package(OMPT)
            if (OMPT_FOUND)
                include_directories(${OMPT_INCLUDE_DIRS})
                set(LIBS ${LIBS} ${OMPT_LIBRARIES})
                if (NOT BUILD_STATIC_EXECUTABLES)
                    set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OMPT_LIBRARY_DIR})
                endif()
            else()
                add_custom_target(project_ompt)
            endif()
        else()
            add_custom_target(project_ompt)
        endif()
    endif()
else()
    add_custom_target(project_ompt)
endif ((NOT DEFINED USE_OMPT) OR (USE_OMPT) OR (BUILD_OMPT))

# Just in case, to prevent concurrent builds
if(APEX_INTEL_MIC)
	add_dependencies (project_ompt project_boost)
endif()

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

if((DEFINED JEMALLOC_ROOT) OR (USE_JEMALLOC))
    find_package(JEmalloc)
    if (JEmalloc_FOUND)
        include_directories(${JEmalloc_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${JEmalloc_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${JEmalloc_LIBRARY_DIR})
        endif()
    endif()
endif()

if((DEFINED GPERFTOOLS_ROOT) OR (USE_TCMALLOC))
    find_package(Tcmalloc)
    if (Tcmalloc_FOUND)
        include_directories(${Tcmalloc_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${Tcmalloc_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${Tcmalloc_LIBRARY_DIR})
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    endif()
endif()

################################################################################
# LM Sensors configuration
################################################################################

if((DEFINED LM_SENSORS_ROOT) OR (USE_LM_SENSORS))
    find_package(LMSensors)
    if (LM_SENSORS_FOUND)
        include_directories(${LM_SENSORS_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${LM_SENSORS_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${LM_SENSORS_LIBRARY_DIR})
        endif()
    endif()
endif()

################################################################################
# ActiveHarmony configuration
################################################################################

if((DEFINED USE_ACTIVEHARMONY) AND (NOT USE_ACTIVEHARMONY))
    # just in case, disable all Active Harmony
    unset(ACTIVEHARMONY_ROOT)
    unset(ENV{ACTIVEHARMONY_ROOT})
    unset(BUILD_ACTIVEHARMONY)
endif()

if(((DEFINED ACTIVEHARMONY_ROOT) OR (BUILD_ACTIVEHARMONY)) OR (USE_ACTIVEHARMONY))
    find_package(ActiveHarmony)
    if (ACTIVEHARMONY_FOUND)
        include_directories(${ACTIVEHARMONY_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${ACTIVEHARMONY_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ACTIVEHARMONY_LIBRARY_DIR})
        endif()
    else()
        add_custom_target(project_activeharmony)
    endif()
else()
    add_custom_target(project_activeharmony)
endif()
# Just in case, to prevent concurrent builds
add_dependencies (project_activeharmony project_ompt)

################################################################################
# PAPI configuration
################################################################################

if((DEFINED PAPI_ROOT) OR (USE_PAPI))
    find_package(PAPI)
    if (PAPI_FOUND)
        include_directories(${PAPI_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${PAPI_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${PAPI_LIBRARY_DIR})
        endif()
    endif()
endif()

################################################################################
# OTF2 configuration
################################################################################

if((DEFINED USE_OTF2) AND (NOT USE_OTF2))
    # just in case, disable
    unset(OTF2_ROOT)
    unset(ENV{OTF2_ROOT})
    unset(BUILD_OTF2)
endif()

if((DEFINED OTF2_ROOT) OR (USE_OTF2) OR (BUILD_OTF2))
    find_package(OTF2)
    if (OTF2_FOUND)
        include_directories(${OTF2_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${OTF2_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OTF2_LIBRARY_DIR})
        endif()
        # add_definitions(-DAPEX_USE_CLOCK_TIMESTAMP=1)
        if(NOT APPLE)
            find_library(RTLIB rt)
            set(LIBS ${LIBS} ${RTLIB})
        endif(NOT APPLE)
    else()
        add_custom_target(project_otf2)
    endif()
else()
    add_custom_target(project_otf2)
endif()
# Just in case, to prevent concurrent builds
add_dependencies (project_otf2 project_activeharmony)

################################################################################
# CUDA configuration
################################################################################

if (APEX_USE_CUDA)
    set(LIBS ${LIBS} cuda ${CUDA_cupti_LIBRARY})
    add_definitions(-DAPEX_USE_CUDA)
endif (APEX_USE_CUDA)

################################################################################
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

if((DEFINED BFD_ROOT) OR (USE_BFD) OR (BUILD_BFD))
    find_package(BFD)
    if (BFD_FOUND)
        include_directories(${BFD_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${BFD_LIBRARIES})
        if (NOT BUILD_STATIC_EXECUTABLES)
            set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${BFD_LIBRARY_DIR})
        endif()
        message(INFO " Using binutils: ${BFD_LIBRARY_DIR} ${BFD_LIBRARIES}")
    endif()

    find_package(Demangle)
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

if(USE_PLUGINS)
    message(INFO " apex will be built with plugin support.")
    set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
    add_definitions("-DAPEX_USE_PLUGINS")

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
    find_package (Threads REQUIRED)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
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
    SET(extralibs_rpath "-Wl${TMP_RPATH}")
endif()
#SET(extralibs "${TMP_LINKFLAGS} ${TMP_LIBS} ${CMAKE_EXE_LINKER_FLAGS}")
SET(extralibs "${TMP_LINKFLAGS} ${TMP_LIBS}")
if (RCR_FOUND)
SET(extraincludes "-I${RCR_INCLUDE_DIR}")
endif (RCR_FOUND)
SET(requirements "")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/pkgconfig/apex.pc.in
               ${CMAKE_BINARY_DIR}/pkgconfig/apex.pc @ONLY)
INSTALL_FILES(/lib/pkgconfig FILES pkgconfig/apex.pc)

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
add_subdirectory (src/apex)
add_subdirectory (src/ITTNotify)
add_subdirectory (src/comm)
if (NOT BUILD_STATIC_EXECUTABLES)
    add_subdirectory (src/wrappers)
endif()
if(MPI_CXX_FOUND)
  add_subdirectory (src/comm/mpi)
endif()
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

if(BUILD_TESTS)
  message(INFO " Building unit tests.")
  if(OMPT_FOUND)
    add_subdirectory (src/openmp)
  endif(OMPT_FOUND)
  add_subdirectory (src/unit_tests/C)
  add_subdirectory (src/unit_tests/C++)
  add_subdirectory (src/unit_tests/CUDA)
else()
    #add_subdirectory (src/unit_tests/C EXCLUDE_FROM_ALL)
    #add_subdirectory (src/unit_tests/C++ EXCLUDE_FROM_ALL)
endif(BUILD_TESTS)

if(BUILD_EXAMPLES)
  message(INFO " Building examples.")
  add_subdirectory (src/examples)
else()
    #add_subdirectory (src/examples EXCLUDE_FROM_ALL)
endif(BUILD_EXAMPLES)

