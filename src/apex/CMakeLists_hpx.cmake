#  Copyright (c) 2014-2021 Kevin Huck
#  Copyright (c) 2014-2021 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Create a library called "apex"

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
cmake_policy(VERSION 2.8.12)
if (${CMAKE_MAJOR_VERSION} GREATER 2)
  cmake_policy(SET CMP0042 NEW)
    if (${CMAKE_MINOR_VERSION} GREATER 11)
      cmake_policy(SET CMP0074 NEW)
    endif()
endif()

hpx_info("apex" "Will build APEX")

set (APEX_VERSION_MAJOR 2)
set (APEX_VERSION_MINOR 7)
set (APEX_VERSION_PATCH 1)

if (NOT APEX_ROOT)
  if (EXISTS ${HPX_SOURCE_DIR}/apex)
    set(APEX_ROOT ${HPX_SOURCE_DIR}/apex)
  else()
    hpx_info("Apex not found")
  endif()
endif()

# Module path both apex and hpx
list(APPEND CMAKE_MODULE_PATH "${APEX_ROOT}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(APEX_DefaultOptions)

set(APEX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APEX_SOURCE_DIR ${APEX_SOURCE_DIR} PARENT_SCOPE)
set(APEX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(APEX_BINARY_DIR ${APEX_BINARY_DIR} PARENT_SCOPE)

hpx_info("apex" "APEX source dir is ${APEX_SOURCE_DIR}")
hpx_info("apex" "APEX binary dir is ${APEX_BINARY_DIR}")

# Add a target to store the flags
add_library(apex_flags INTERFACE)

# This macro will make sure that the hpx/config.h file is included
target_compile_definitions(apex_flags INTERFACE APEX_HAVE_HPX_CONFIG)
# This macro will be added to the hpx/config.h file.
hpx_add_config_define(APEX_HAVE_HPX)   # tell HPX that we use APEX

if(APEX_DEBUG)
  target_compile_definitions(apex_flags INTERFACE APEX_DEBUG)
endif()

# Check if architecture is x86 or not
message("System architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set(APEX_ARCH_X86 TRUE)
else()
  set(APEX_ARCH_X86 FALSE)
endif()

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

site_name(HOSTNAME)
string(LENGTH ${HOSTNAME} HOSTNAME_LENGTH)
if(${HOSTNAME_LENGTH} GREATER 5)
    string(SUBSTRING ${HOSTNAME} 0 6 HOST_BASENAME)
else()
    set (HOST_BASENAME ${HOSTNAME})
endif()

if(NOT MSVC)
  target_compile_options(apex_flags INTERFACE -fPIC)
else()
  target_compile_definitions(apex_flags INTERFACE -D_WINDOWS)
  target_compile_definitions(apex_flags INTERFACE -D_WIN32)
  target_compile_definitions(apex_flags INTERFACE -D_WIN32_WINNT=0x0601)
  hpx_add_compile_flag(-wd4800)     # forcing value to bool 'true' or 'false' (performance warning)
  hpx_add_compile_flag(-wd4244)     # conversion from '...' to '...', possible loss of data
  hpx_add_compile_flag(-wd4267)     # conversion from '...' to '...', possible loss of data

  # VS2012 and above has a special flag for improving the debug experience by
  # adding more symbol information to the build (-d2Zi+)
  hpx_add_compile_flag(-d2Zi+ CONFIGURATIONS RelWithDebInfo)

  # VS2013 and above know how to do link time constant data segment folding
  # VS2013 update 2 and above know how to remove debug information for
  #     non-referenced functions and data (-Zc:inline)
  if(MSVC12 OR MSVC14)
    hpx_add_compile_flag(-Zc:inline)
    hpx_add_compile_flag(-Gw CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)
    hpx_add_compile_flag(-Zo CONFIGURATIONS RelWithDebInfo)
  endif()

  if(MSVC14)
    # assume conforming (throwing) operator new implementations
    hpx_add_target_compile_option(-Zc:throwingNew)

    # Update 2 requires to set _ENABLE_ATOMIC_ALIGNMENT_FIX for it to compile
    # atomics
    hpx_add_config_define(_ENABLE_ATOMIC_ALIGNMENT_FIX)

    # Update 3 allows to flag rvalue misuses and enforces strict string const-
    # qualification conformance
    hpx_add_target_compile_option(-Zc:rvalueCast)
    hpx_add_target_compile_option(-Zc:strictStrings)
  endif()

  hpx_add_compile_flag(-bigobj) # Increase the maximum size of object file sections
  hpx_add_compile_flag(-MP)     # Multiprocessor build
endif()

# Proc
if(EXISTS "/proc/stat" AND NOT APEX_DISABLE_PROC)
  hpx_info("apex" "Building APEX with /proc/stat sampler")
  set(APEX_HAVE_PROC TRUE)
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_PROC)
  set(proc_headers proc_read.h)
  set(proc_sources proc_read.cpp)
endif()

IF("$ENV{CRAY_CPU_TARGET}" STREQUAL "mic-knl")
    hpx_info("apex" "This is Cray KNL, will build with Cray Power support")
    set(APEX_HAVE_CRAY_POWER TRUE)
    target_compile_definitions(apex_flags INTERFACE APEX_HAVE_CRAY_POWER)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  hpx_info("apex" "APEX is being built with a Debug configuration")
  target_compile_definitions(apex_flags INTERFACE DEBUG)
  target_compile_definitions(apex_flags INTERFACE -DCMAKE_BUILD_TYPE=3)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  hpx_info("apex" "APEX is being built with a RelWithDebInfo configuration")
  target_compile_definitions(apex_flags INTERFACE -DCMAKE_BUILD_TYPE=2)
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  hpx_info("apex" "APEX is being built with a Release configuration")
  target_compile_definitions(apex_flags INTERFACE -DCMAKE_BUILD_TYPE=1)
endif()


#######################################################
# We create one target per imported library (cleaner) #
set(_apex_imported_targets "")

include(APEX_SetupCUPTI)
if(APEX_WITH_CUDA)
    set(cupti_sources cupti_trace.cpp)
    set(nvml_sources apex_nvml.cpp)
endif(APEX_WITH_CUDA)

include(APEX_SetupHIP)
if(APEX_WITH_HIP)
    if (ROCPROFILER_FOUND)
    SET(ROCPROFILER_SOURCE
        hip_profiler.cpp
        ctrl/test_hsa.cpp
        util/hsa_rsrc_factory.cpp
        util/perf_timer.cpp)
    endif(ROCPROFILER_FOUND)

    set(hip_sources hip_trace.cpp apex_rocm_smi.cpp ${ROCPROFILER_SOURCE})
endif(APEX_WITH_HIP)

# Setup Kokkos
include(APEX_SetupKokkos)

# Setup PAPI
include(APEX_SetupPAPI)

# Setup ZLIB
include(APEX_SetupZlib)

# Setup LM Sensors
include(APEX_SetupLMSensors)
if(APEX_WITH_LM_SENSORS)
  set(sensor_sources sensor_data.cpp)
endif()

# Setup Active Harmony
include(APEX_SetupActiveHarmony)

# Currently setup Rapidjson
include(APEX_SetupPlugins)

if((DEFINED RCR_ROOT) OR (APEX_WITH_RCR) OR (APEX_BUILD_RCR))
	find_package(RCR)
endif()

# RCR
if(HAVE_RCR)
	hpx_add_config_define(HPX_HAVE_RCR 1)
else()
  if(("${HOST_BASENAME}" STREQUAL "edison") OR ("$ENV{NERSC_HOST}" STREQUAL "edison") OR
    ("${HOST_BASENAME}" STREQUAL "cori") OR ("$ENV{NERSC_HOST}" STREQUAL "cori"))
    #add_definitions(-fPIC)
    set (APEX_HAVE_CRAY_POWER TRUE)
    target_compile_definitions(apex_flags INTERFACE APEX_HAVE_CRAY_POWER)
    message(INFO " System has Cray energy monitoring support.")
  else()
    if(EXISTS "/sys/class/powercap/intel-rapl/intel-rapl:0")
      set (APEX_HAVE_POWERCAP_POWER TRUE)
      target_compile_definitions(apex_flags INTERFACE APEX_HAVE_POWERCAP_POWER)
      message(INFO " System has Powercap energy monitoring support.")
    endif()
  endif()
endif()

# Setup BFD
if((DEFINED BFD_ROOT) OR (APEX_WITH_BFD) OR (APEX_BUILD_BFD))
  set(APEX_WITH_BFD ${APEX_WITH_BFD})
  set(APEX_BUILD_BFD ${APEX_BUILD_BFD})
  # Setup BFD
  include(APEX_SetupBFD)
  set(bfd_sources apex_bfd.cpp address_resolution.cpp)
  # Setup DEMANGLE
  include(APEX_SetupDemangle)
  # Setup Zlib - already done for google trace events output
#	if(NOT APEX_INTEL_MIC)
#    include(APEX_SetupZlib)
#	endif(NOT APEX_INTEL_MIC)
else()
    set(bfd_sources address_resolution.cpp)
	add_custom_target(project_binutils)
endif()

# Setup MSR
include(APEX_SetupMSR)

# Setup OTF2
include(APEX_SetupOTF2)
if(APEX_WITH_OTF2)
  set(otf2_headers otf2_listener.hpp)
  set(otf2_sources otf2_listener.cpp otf2_listener_hpx.cpp otf2_listener_nompi.cpp)
endif()

if(APEX_WITH_PERFETTO)
  set(perfetto_headers perfetto_listener.hpp perfetto_static.hpp)
  set(perfetto_sources
    perfetto_listener.cpp perfetto_static.cpp ../perfetto_sdk/perfetto.cc)
  target_compile_definitions(apex_flags INTERFACE APEX_WITH_PERFETTO)
endif()

if(HPX_WITH_HPXMP)
  set(ompt_sources apex_ompt.cpp)
  include(APEX_SetupHPXMP)
  set(ompt_dependency apex::hpxmp)
  target_compile_definitions(apex_flags INTERFACE APEX_WITH_OMPT)
  target_compile_definitions(apex_flags INTERFACE ${OpenMP_CXX_FLAGS})
endif(HPX_WITH_HPXMP)

# Setup MPI
include(APEX_SetupMPI)

#if(HPX_WITH_PARCELPORT_MPI)
  #set(apex_mpi_sources apex_mpi.cpp)
#endif(HPX_WITH_PARCELPORT_MPI)


# RPATH related stuff
# APEX is always a shared object library with HPX now.
# So set up RPATH regardless, especially for OTF, PAPI, AH, etc.
# 1) use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# 2) when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# 3) add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Everything else RPATH-related is handled by HPX...

##################################
# Setup main apex library #
set(apex_headers
    apex.h
    apex.hpp
    apex_api.hpp
    apex_assert.h
    apex_clock.hpp
    apex_cxx_shared_lock.hpp
    apex_export.h
    apex_api.hpp
    apex_kokkos.hpp
    apex_options.hpp
    apex_policies.hpp
    apex_types.h
    concurrency_handler.hpp
    csv_parser.h
    dependency_tree.hpp
    event_listener.hpp
    exhaustive.hpp
    genetic_search.hpp
    gzstream.hpp
    handler.hpp
    memory_wrapper.hpp
    nelder_mead.hpp
    policy_handler.hpp
    profile.hpp
    profiler.hpp
    profile_reducer.hpp
    profiler_listener.hpp
    random.hpp
    semaphore.hpp
    simulated_annealing.hpp
    thread_instance.hpp
    threadpool.h
    task_identifier.hpp
    task_wrapper.hpp
    tau_listener.hpp
    tree.h
    utils.hpp
    ${perfetto_headers}
    ${proc_headers}
    ${otf2_headers}
    )
    #apex_config.h

set(apex_sources
    apex.cpp
    apex_dynamic.cpp
    apex_error_handling.cpp
    apex_kokkos.cpp
    apex_kokkos_tuning.cpp
    ${apex_mpi_sources}
    apex_options.cpp
    apex_policies.cpp
    concurrency_handler.cpp
    csv_parser.cpp
    dependency_tree.cpp
    event_listener.cpp
    event_filter.cpp
    exhaustive.cpp
    genetic_search.cpp
    gzstream.cpp
    handler.cpp
    memory_wrapper.cpp
    nelder_mead.cpp
    nvtx_listener.cpp
    policy_handler.cpp
    profile_reducer.cpp
    profiler_listener.cpp
    random.cpp
    simulated_annealing.cpp
    task_identifier.cpp
    tau_listener.cpp
    tau_dummy.cpp
    thread_instance.cpp
    threadpool.cpp
    trace_event_listener.cpp
    tree.cpp
    utils.cpp
    ${perfetto_sources}
    ${proc_sources}
    ${bfd_sources}
    ${sensor_sources}
    ${otf2_sources}
    ${ompt_sources}
    ${cupti_sources}
    ${nvml_sources}
    ${hip_sources}
    )

include(GNUInstallDirs)

if (WIN32)
# Enable standards-compliant mode when using the Visual Studio compiler.
    if (MSVC)
        SET_SOURCE_FILES_PROPERTIES( perfetto.cc PROPERTIES COMPILE_FLAGS
            "/permissive- /bigobj -DWIN32_LEAN_AND_MEAN -DNOMINMAX" )
    else (MSVC)
        SET_SOURCE_FILES_PROPERTIES( perfetto.cc PROPERTIES COMPILE_FLAGS
            "/bigobj -DWIN32_LEAN_AND_MEAN -DNOMINMAX" )
    endif (MSVC)
endif (WIN32)

# Force C++11 with NVHPC, it crashes otherwise
if (CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    set(CMAKE_CXX_STANDARD 11)
endif()

# APEX has one dependency in HPX, just the main library.
# Unless HPXMP is also used.

add_hpx_library(apex
  DYNAMIC
  INSTALL
  SOURCES ${apex_sources}
  HEADERS ${apex_headers}
  DEPENDENCIES
    ${_dependencies}
    ${ompt_dependency}
  FOLDER "Core/Dependencies")

target_include_directories(apex PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<BUILD_INTERFACE:${APEX_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../perfetto_sdk>
    $<INSTALL_INTERFACE:include>)

# To have the compile options and definitions
target_link_libraries(apex PRIVATE apex_flags)

# Link to all imported targets
hpx_info("apex" "Will build APEX with imported targets:")
foreach(lib IN LISTS _apex_imported_targets)
  if (TARGET "${lib}")
    hpx_info("apex" "    ${lib}")
    target_link_libraries(apex PRIVATE ${lib})
  endif()
endforeach()

# Link to dl
if (APEX_USE_WEAK_SYMBOLS OR MSVC)
  target_compile_definitions(apex_flags INTERFACE APEX_USE_WEAK_SYMBOLS)
else()
  find_library(DYNAMICLIB dl)
  target_link_libraries(apex PRIVATE ${DYNAMICLIB})
endif (APEX_USE_WEAK_SYMBOLS OR MSVC)

# TODO: see how to remove those global include directories
# add the binary tree to the search path for include files
# so that we will find apex_config.h
if(HAVE_RCR)
  target_include_directories(apex PUBLIC RCR_INCLUDE_PATH})
endif()

configure_file (
  "${APEX_SOURCE_DIR}/apex_config.h.in"
  "${APEX_BINARY_DIR}/apex_config.h")

# Needed to configure apex_exec
set(APEX_LIBRARY_NAME libhpx_apex)
CONFIGURE_FILE(${APEX_SOURCE_DIR}/../scripts/apex_exec
	${APEX_BINARY_DIR}/apex_exec @ONLY)
INSTALL(FILES ${APEX_BINARY_DIR}/apex_exec DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)

install(FILES ${apex_headers} DESTINATION include)
install(FILES "${APEX_BINARY_DIR}/apex_config.h" DESTINATION include)

hpx_export_targets(apex_flags)
install(TARGETS apex_flags EXPORT HPXTargets)

set(AMPLIFIER_ROOT ${AMPLIFIER_ROOT} PARENT_SCOPE)
set(APEX_FOUND ON PARENT_SCOPE)

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

message(STATUS "----------------------------------------------------------------------")
message(STATUS "APEX Variable Report:")
message(STATUS "----------------------------------------------------------------------")
dump_cmake_variables("^APEX")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Libraries: " ${LIBS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "Install Prefix:" ${CMAKE_INSTALL_PREFIX})
message(STATUS "----------------------------------------------------------------------")

