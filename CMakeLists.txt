# The name of our project is "APEX". CMakeLists files in this project can
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and
# to the root binary directory of the project as ${APEX_BINARY_DIR}.

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
cmake_policy(VERSION 2.8.12)
if (${CMAKE_MAJOR_VERSION} GREATER 2)
  cmake_policy(SET CMP0042 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

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

# set the project name
project (APEX CXX C)

# If TAU is used, don't allow throttling - it can potentially lead to
# overlapping timer errors in TAU.
if((DEFINED APEX_THROTTLE) AND (APEX_THROTTLE))
    if ((NOT DEFINED TAU_ROOT) AND (NOT USE_TAU))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_THROTTLE")
    endif()
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

set(NORMAL_COMPILER_WARNINGS "-Wextra -Waddress -Warray-bounds -Wuninitialized -Wreturn-type -Wall -fno-omit-frame-pointer ")

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

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG -Wno-deprecated")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
  set(CXX_PEDANTIC_COMPILER_WARNINGS "-fdiagnostics-show-option -Wall -Wcast-align -Wcast-qual -Werror=format -Werror=missing-braces -Werror=parentheses -Werror=reorder -Werror=return-type -Werror=sequence-point -Werror=sign-compare -Werror=uninitialized -Werror=vla -Wextra -Wformat=2 -Winit-self -Wno-attributes -Wno-cast-align -Wno-delete-non-virtual-dtor -Wno-format-nonliteral -Wno-ignored-qualifiers -Wno-sign-promo -Wno-strict-aliasing -Wno-unused-parameter -Wno-unused-variable -Wno-unused-local-typedefs -Wno-unused-parameter ")
  set(C_PEDANTIC_COMPILER_WARNINGS "-fdiagnostics-show-option -Wall -Wcast-align -Wcast-qual -Werror=format -Werror=missing-braces -Werror=parentheses -Werror=reorder -Werror=return-type -Werror=sequence-point -Werror=sign-compare -Werror=uninitialized -Werror=vla -Wextra -Wformat=2 -Winit-self -Wno-attributes -Wno-cast-align -Wno-format-nonliteral -Wno-ignored-qualifiers -Wno-strict-aliasing -Wno-unused-parameter -Wno-unused-variable -Wno-unused-local-typedefs -Wno-unused-parameter ")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CXX_PEDANTIC_COMPILER_WARNINGS} -DAPEX_DEBUG")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0 ${C_PEDANTIC_COMPILER_WARNINGS} -DAPEX_DEBUG")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
     message(FATAL_ERROR "GCC version must be at least 4.9!")
  endif()

  set(CXX_PEDANTIC_COMPILER_WARNINGS "-fdiagnostics-show-option -Wall -Wcast-align -Wcast-qual -Wdouble-promotion -Werror=format -Werror=missing-braces -Werror=parentheses -Werror=reorder -Werror=return-type -Werror=sequence-point -Werror=sign-compare -Werror=trampolines -Werror=uninitialized -Werror=vla -Wextra -Wformat=2 -Winit-self -Wno-attributes -Wno-cast-align -Wno-delete-non-virtual-dtor -Wno-format-nonliteral -Wno-ignored-qualifiers -Wno-sign-promo -Wno-strict-aliasing -Wno-sync-nand -Wno-unused-but-set-parameter -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wno-unused-parameter ")
  set(C_PEDANTIC_COMPILER_WARNINGS "-fdiagnostics-show-option -Wall -Wcast-align -Wcast-qual -Wdouble-promotion -Werror=format -Werror=missing-braces -Werror=parentheses -Werror=reorder -Werror=return-type -Werror=sequence-point -Werror=sign-compare -Werror=trampolines -Werror=uninitialized -Werror=vla -Wextra -Wformat=2 -Winit-self -Wno-attributes -Wno-cast-align -Wno-format-nonliteral -Wno-ignored-qualifiers -Wno-strict-aliasing -Wno-sync-nand -Wno-unused-but-set-parameter -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wno-unused-parameter ")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CXX_PEDANTIC_COMPILER_WARNINGS} ${APEX_SANITIZE_OPTIONS} -DAPEX_DEBUG")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0 ${C_PEDANTIC_COMPILER_WARNINGS} ${APEX_SANITIZE_OPTIONS} -DAPEX_DEBUG")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DAPEX_DEBUG")
  set(CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

# The version number.
set (APEX_VERSION_MAJOR 0)
set (APEX_VERSION_MINOR 5)

# add_definitions(-std=c++11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    else()
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
        if(COMPILER_SUPPORTS_CXX1Y)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
        else()
            CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
            if(COMPILER_SUPPORTS_CXX11)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
            else()
                CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
                if(COMPILER_SUPPORTS_CXX0X)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
                else()
                    message(FATAL_ERROR " Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
                endif()
            endif()
        endif()
    endif()
endif()

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

################################################################################
# TAU configuration
################################################################################

# TAU is NOT required.
if ((DEFINED TAU_ROOT) OR (USE_TAU))
find_package(TAU)
if (TAU_FOUND)
    include_directories(${TAU_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${TAU_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${TAU_LIBRARY_DIR})
    endif()
    set(USE_TAU TRUE)
endif()
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

if((DEFINED USE_OMPT) AND (NOT USE_OMPT))
    # just in case, disable all ompt
    unset(OMPT_ROOT)
    unset(ENV{OMPT_ROOT})
    unset(BUILD_OMPT)
endif()

if ((NOT DEFINED USE_OMPT) OR (USE_OMPT) OR (BUILD_OMPT))
    find_package(OpenMP)
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
        add_definitions(-DAPEX_USE_CLOCK_TIMESTAMP=1)  
        find_library(RTLIB rt)
        set(LIBS ${LIBS} ${RTLIB})
    else()
        add_custom_target(project_otf2)
    endif()
else()
    add_custom_target(project_otf2)
endif()

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

################################################################################
# OCR configuration
################################################################################

if((DEFINED OCR_ROOT) OR (USE_OCR))
    find_package(OCR)
    if(NOT OCR_FOUND)
        message(FATAL_ERROR " Requested USE_OCR but could not find OCR; set OCR_ROOT")
    endif()
    message(INFO " Using OCR: ${OCR_LIBRARY_DIR} ${OCR_LIBRARIES} ${OCR_INCLUDE_DIRS}")
    include_directories(${OCR_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OCR_LIBRARIES})
    if (NOT BUILD_STATIC_EXECUTABLES)
        set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OCR_LIBRARY_DIR})
    endif()
    if(USE_OCR_LOAD_BALANCE)
        set(USE_LOAD_BALANCE TRUE)
        message(INFO " apex will be built with OCR load balancing support.")
        add_definitions("-DAPEX_OCR_LOAD_BALANCE")
    endif()
endif()

################################################################################
# Plugins support
################################################################################

if(USE_PLUGINS)
    message(INFO " apex will be built with plugin support.")
    set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
    add_definitions("-DAPEX_USE_PLUGINS")
endif()

################################################################################
# Load balancing support
################################################################################

if(USE_LOAD_BALANCE)
    message(INFO " apex will be built with load balancing support.")
    add_definitions("-DAPEX_USE_LOAD_BALANCE")
endif()

################################################################################
# Standard Library configuration
################################################################################

# checking for pthread the correct way is broken for MIC.
if(APEX_INTEL_MIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
else()
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif(APEX_INTEL_MIC)

if (RCR_FOUND)
    find_library(RTLIB rt)
    set(LIBS ${LIBS} ${RTLIB})
    find_library(STDLIBCPP stdc++)
    set(LIBS ${LIBS} ${STDLIBCPP})
endif()

# apparently, we need to make sure libm is last.
find_library(MATHLIB m)
set(LIBS ${LIBS} ${MATHLIB})
#message(${LIBS})

IF(("${HOST_BASENAME}" STREQUAL "edison") OR ("$ENV{NERSC_HOST}" STREQUAL "edison"))
    if (BFD_FOUND)
        find_library(DYNAMICLIB dl)
        set(LIBS ${LIBS} ${DYNAMICLIB})
    endif()
else()
    if (BUILD_STATIC_EXECUTABLES)
        find_library(DYNAMICLIB dl)
        set(LIBS ${LIBS} ${DYNAMICLIB})
    endif()
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

if(BUILD_TESTS)
  message(INFO " Building unit tests.")
  add_subdirectory (src/unit_tests/C)
  add_subdirectory (src/unit_tests/C++)
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

if(TAU_FOUND)
# set any extra envionment variables here
SET (CTEST_ENVIRONMENT "APEX_TAU=1")
endif(TAU_FOUND)
