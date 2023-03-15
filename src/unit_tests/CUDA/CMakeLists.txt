if(MPI_CXX_FOUND)
    set(APEX_CUDA_CXX_FLAGS ${MPI_COMPILE_FLAGS})
    set(APEX_CUDA_C_FLAGS ${MPI_COMPILE_FLAGS})
    set(APEX_CUDA_EXTRA_INCLUDE ${MPI_CXX_INCLUDE_PATH})
    set(APEX_CUDA_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
    set(APEX_RING_EXAMPLE apex_ring_dtod)
    set(APEX_P2P_EXAMPLE apex_p2p)
    #set(APEX_PING_PONG_EXAMPLE apex_ping_pong)
endif()

# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_CUDA_EXTRA_INCLUDE})
include_directories(${APEX_SOURCE_DIR}/src/unit_tests/CUDA)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

set(example_programs
    apex_cuda
    apex_multi_cuda
    apex_vector
    multiGpuThread
    ${APEX_RING_EXAMPLE}
    ${APEX_P2P_EXAMPLE}
    ${APEX_PING_PONG_EXAMPLE}
   )

if (OPENMP_FOUND)
  set(example_programs "${example_programs};pi")
endif (OPENMP_FOUND)

message(INFO "Using CUDA libraries: ${CUDA_LIBRARIES}")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${APEX_CUDA_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${APEX_CUDA_CXX_FLAGS}")

foreach(example_program ${example_programs})
  set(sources ${example_program}.cu)
  source_group("Source Files" FILES ${sources})
  add_executable("${example_program}_cu" ${sources})
  target_link_libraries ("${example_program}_cu" apex ${LIBS} OpenMP::OpenMP_CXX CUDA::cuda_driver curand nvToolsExt ${APEX_CUDA_CXX_LINK_FLAGS})
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties("${example_program}_cu" PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
  add_dependencies ("${example_program}_cu" apex)
  add_dependencies (tests "${example_program}_cu")
  add_test ("test_${example_program}_cu" "${example_program}_cu")
  set_tests_properties("test_${example_program}_cu" PROPERTIES TIMEOUT 30)
  # install(TARGETS "${example_program}_cu" RUNTIME DESTINATION "bin/apex_unit_tests" OPTIONAL)
  set_tests_properties("test_${example_program}_cu" PROPERTIES ENVIRONMENT "APEX_ENABLE_CUDA=1;APEX_CUDA_KERNEL_DETAILS=1;APEX_CUDA_COUNTERS=1;APEX_CUDA_DRIVER_API=1")

endforeach()

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)


