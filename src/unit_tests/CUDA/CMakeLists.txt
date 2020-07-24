# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

set(example_programs
    apex_cuda
    apex_vector
    multiGpuThread
   )

if (OPENMP_FOUND)
  set(example_programs "${example_programs};pi")
endif (OPENMP_FOUND)

message(INFO "Using CUDA libraries: ${CUDA_LIBRARIES}")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

foreach(example_program ${example_programs})
  set(sources ${example_program}.cu)
  source_group("Source Files" FILES ${sources})
  add_executable("${example_program}_cu" ${sources})
  target_link_libraries ("${example_program}_cu" apex ${LIBS} OpenMP::OpenMP_CXX cuda curand)
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties("${example_program}_cu" PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
  add_dependencies ("${example_program}_cu" apex)
  add_dependencies (tests "${example_program}_cu")
  add_test ("test_${example_program}_cu" "${example_program}_cu")
  # install(TARGETS "${example_program}_cu" RUNTIME DESTINATION "bin/apex_unit_tests" OPTIONAL)
endforeach()

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)


