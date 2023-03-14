#  Copyright (c) 2014-2021 Kevin Huck
#  Copyright (c) 2014-2021 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Create a library called "Apex" which includes the source file "apex.cxx".
# The extension is already found. Any number of sources could be listed here.

include_directories(${APEX_SOURCE_DIR}/src/apex ${PROJECT_BINARY_DIR}/src/apex ${APEX_SOURCE_DIR}/rapidjson/include)

SET(tau_SOURCE tau_listener.cpp)

if (BFD_FOUND)
SET(bfd_SOURCE apex_bfd.cpp address_resolution.cpp)
else(BFD_FOUND)
SET(bfd_SOURCE address_resolution.cpp)
endif(BFD_FOUND)

if (APEX_HAVE_PROC)
SET(PROC_SOURCE proc_read.cpp)
endif(APEX_HAVE_PROC)

if (LM_SENSORS_FOUND)
SET(SENSOR_SOURCE sensor_data.cpp)
endif(LM_SENSORS_FOUND)

if (OTF2_FOUND)
SET(OTF2_SOURCE otf2_listener.cpp otf2_listener_mpi.cpp otf2_listener_nompi.cpp)
endif(OTF2_FOUND)

if (ZLIB_FOUND)
SET(ZLIB_SOURCE gzstream.cpp)
endif(ZLIB_FOUND)

if (APEX_WITH_MPI)
SET(MPI_SOURCE apex_mpi.cpp)
endif (APEX_WITH_MPI)

if (MPI_Fortran_FOUND)
SET(MPIF_SOURCE apex_mpif.F90)
endif (MPI_Fortran_FOUND)

if (STARPU_FOUND)
SET(STARPU_SOURCE apex_starpu.cpp)
endif(STARPU_FOUND)

if (APEX_WITH_PHIPROF)
SET(PHIPROF_SOURCE apex_phiprof.cpp)
endif (APEX_WITH_PHIPROF)

if (OpenACC_CXX_FOUND AND OpenACCProfiling_FOUND)
    set(OpenACC_SOURCE apex_openacc.cpp)
endif (OpenACC_CXX_FOUND AND OpenACCProfiling_FOUND)

if (APEX_WITH_RAJA AND RAJA_FOUND)
    set(RAJA_SOURCE apex_raja.cpp)
endif (APEX_WITH_RAJA AND RAJA_FOUND)

if(APEX_WITH_PERFETTO)
    set(perfetto_headers perfetto_listener.hpp)
    set(perfetto_sources
        perfetto_listener.cpp perfetto_static.cpp)
    set(perfetto_target perfetto)
    add_definitions(-DAPEX_WITH_PERFETTO)
    include_directories(${APEX_SOURCE_DIR}/src/perfetto_sdk)
endif()

# Try to keep this in alphabetical order
SET(all_SOURCE
apex_preload.cpp
apex_dynamic.cpp
apex.cpp
apex_error_handling.cpp
apex_kokkos.cpp
apex_kokkos_tuning.cpp
${MPI_SOURCE}
${MPIF_SOURCE}
apex_options.cpp
event_filter.cpp
apex_policies.cpp
${bfd_SOURCE}
${OpenACC_SOURCE}
${RAJA_SOURCE}
${STARPU_SOURCE}
${PHIPROF_SOURCE}
concurrency_handler.cpp
dependency_tree.cpp
event_listener.cpp
exhaustive.cpp
handler.cpp
memory_wrapper.cpp
${OTF2_SOURCE}
${perfetto_sources}
perftool_implementation.cpp
policy_handler.cpp
${PROC_SOURCE}
profile_reducer.cpp
profiler_listener.cpp
random.cpp
${SENSOR_SOURCE}
simulated_annealing.cpp
task_identifier.cpp
tcmalloc_hooks.cpp
${tau_SOURCE}
thread_instance.cpp
trace_event_listener.cpp
utils.cpp
${ZLIB_SOURCE}
)

add_library (apex ${all_SOURCE})

if (OMPT_FOUND)
    add_definitions(-DAPEX_WITH_OMPT)
    SET(OMPT_SOURCE apex_ompt.cpp)
    add_library (apex_ompt ${OMPT_SOURCE})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
        # Use the tool-enabled OpenMP runtime
        target_compile_options(apex_ompt PRIVATE "-mp=ompt")
        target_link_options(apex_ompt PRIVATE "-mp=ompt")
    else()
        target_link_libraries (apex_ompt OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries (apex_ompt apex ${LIBS})
    add_dependencies (apex_ompt apex)
endif (OMPT_FOUND)

add_library (taudummy tau_dummy.cpp)
add_dependencies (apex taudummy ${perfetto_target})

if (APEX_WITH_CUDA)
    add_definitions(-DAPEX_WITH_CUDA)
    include_directories (${CUDAToolkit_INCLUDE_DIR})
    if (CUPTI_FOUND)
        SET(CUPTI_SOURCE cupti_trace.cpp)
    endif(CUPTI_FOUND)
    if (NVML_FOUND)
        SET(NVML_SOURCE apex_nvml.cpp)
    endif(NVML_FOUND)
    add_library (apex_cuda
        ${CUPTI_SOURCE}
        ${NVML_SOURCE})
    target_link_libraries (apex_cuda apex ${LIBS}
        ${CUPTI_LIBRARIES}
        ${NVML_LIBRARIES})
    add_dependencies (apex_cuda apex)
endif (APEX_WITH_CUDA)

if (APEX_WITH_HIP)
    add_definitions(-DAPEX_WITH_HIP)
    if (ROCTRACER_FOUND)
        SET(ROCTRACER_SOURCE hip_trace.cpp)
    endif(ROCTRACER_FOUND)
    if (ROCPROFILER_FOUND)
        SET(ROCPROFILER_SOURCE
            hip_profiler.cpp
            ctrl/test_hsa.cpp
            util/hsa_rsrc_factory.cpp
            util/perf_timer.cpp)
    endif(ROCPROFILER_FOUND)
    if (RSMI_FOUND)
        SET(RSMI_SOURCE apex_rocm_smi.cpp)
    endif(RSMI_FOUND)
    add_library (apex_hip
        ${ROCTRACER_SOURCE}
        ${ROCPROFILER_SOURCE}
        ${RSMI_SOURCE})
    target_link_libraries (apex_hip apex ${LIBS}
        ${ROCTRACER_LIBRARIES}
        ${ROCPROFILER_LIBRARIES}
        ${ROCTX_LIBRARIES}
        ${RSMI_LIBRARIES})
    add_dependencies (apex_hip apex)
endif (APEX_WITH_HIP)

if (APEX_WITH_LEVEL0)
    SET(LEVEL0_SOURCE apex_level0.cpp)
    add_definitions(-DAPEX_WITH_LEVEL0)
    add_library (apex_level0
        ${LEVEL0_SOURCE})
    target_link_libraries (apex_level0 apex ${LIBS}
        ${LEVEL0_LIBRARIES})
    add_dependencies (apex_level0 apex)
endif (APEX_WITH_LEVEL0)

if(ACTIVEHARMONY_FOUND)
    add_dependencies (apex project_activeharmony)
endif(ACTIVEHARMONY_FOUND)

if(OMPT_FOUND)
    add_dependencies (apex project_ompt)
endif(OMPT_FOUND)

if(BFD_FOUND)
    add_dependencies (apex project_binutils)
endif(BFD_FOUND)

if(OTF2_FOUND)
    add_dependencies (apex project_otf2)
endif(OTF2_FOUND)

if(APEX_INTEL_MIC)
    add_dependencies (apex project_boost)
endif(APEX_INTEL_MIC)

if (MPI_CXX_FOUND)
  include_directories (${MPI_CXX_INCLUDE_PATH})
endif(MPI_CXX_FOUND)

if (MPI_Fortran_FOUND)
  include_directories (${MPI_Fortran_INCLUDE_PATH})
endif(MPI_Fortran_FOUND)

# If we are building libapex.so, we want to include all the other libraries,
# so that we can LD_PRELOAD this library with all requirements met.
if (NOT BUILD_STATIC_EXECUTABLES)
    if(APPLE)
        target_link_libraries(apex ${perfetto_target} ${LIBS})
        set_target_properties(apex PROPERTIES LINK_FLAGS "${CMAKE_CURRENT_BINARY_DIR}/libtaudummy.dylib -flat_namespace")
    else(APPLE)
        target_link_libraries(apex ${LIBS} ${perfetto_target} taudummy)
    endif(APPLE)
endif()

if (APEX_WITH_HIP)
    include_directories(${ROCM_ROOT}/include ${ROCM_ROOT}/include/hsa)
endif (APEX_WITH_HIP)

# add the binary tree to the search path for include files
# so that we will find ApexConfig.h
if(HAVE_RCR)
include_directories("${PROJECT_BINARY_DIR}/src/apex" ${RCR_INCLUDE_PATH})
else()
include_directories("${PROJECT_BINARY_DIR}/src/apex")
endif()

SET(APEX_PUBLIC_HEADERS apex.h
    apex_api.hpp
    apex_assert.h
    apex_clock.hpp
    apex_types.h
    apex_policies.h
    apex_policies.hpp
    exhaustive.hpp
    dependency_tree.hpp
    handler.hpp
    memory_wrapper.hpp
    profile.hpp
    random.hpp
    apex_export.h
    utils.hpp
    apex_options.hpp
    profiler.hpp
    simulated_annealing.hpp
    task_wrapper.hpp
    task_identifier.hpp)

INSTALL(FILES ${APEX_PUBLIC_HEADERS} DESTINATION include)
#set_target_properties(apex PROPERTIES PUBLIC_HEADER apex.h)

INSTALL(TARGETS apex EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)
if (APEX_WITH_OMPT)
INSTALL(TARGETS apex_ompt EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)
endif (APEX_WITH_OMPT)
if (APEX_WITH_HIP)
INSTALL(TARGETS apex_hip EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)
endif (APEX_WITH_HIP)
if (APEX_WITH_CUDA)
INSTALL(TARGETS apex_cuda EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)
endif (APEX_WITH_CUDA)
if (APEX_WITH_LEVEL0)
INSTALL(TARGETS apex_level0 EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)
endif (APEX_WITH_LEVEL0)
INSTALL(TARGETS taudummy EXPORT APEXTargets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)

install(EXPORT APEXTargets FILE APEXTargets.cmake NAMESPACE APEX:: DESTINATION lib/cmake/APEX)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/APEXConfig.cmake
    INSTALL_DESTINATION lib/cmake/APEX
    )

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/APEXConfig.cmake
    DESTINATION lib/cmake/APEX
    )

