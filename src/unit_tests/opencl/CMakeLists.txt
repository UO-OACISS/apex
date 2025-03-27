# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

if(OpenCL_FOUND)

set(example_programs
    opencl_matmult
   )

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)
  source_group("Source Files" FILES ${sources})
  add_executable("${example_program}_cpp" ${sources})
  target_link_libraries ("${example_program}_cpp" apex ${LIBS} OpenCL::OpenCL)
  # This is needed to make sure local symbols are exported and we can dladdr them
  set_property(TARGET "${example_program}_cpp" PROPERTY ENABLE_EXPORTS ON)
  add_dependencies ("${example_program}_cpp" apex)
  add_dependencies (tests "${example_program}_cpp")
  add_test ("test_${example_program}_cpp" "${example_program}_cpp")
  set_tests_properties("test_${example_program}_cpp" PROPERTIES TIMEOUT 30)
endforeach()

# Make sure the compiler can find include files from our Apex library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${MPI_CXX_INCLUDE_PATH})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

endif(OpenCL_FOUND)
