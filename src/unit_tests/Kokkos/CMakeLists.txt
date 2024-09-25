# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)
include_directories (${APEX_SOURCE_DIR}/src/unit_tests/Kokkos)
#include_directories(${Kokkos_INCLUDE_DIRS_RET})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

if(NOT APPLE)
    set(openmp_programs mm2d_tiling)
endif(NOT APPLE)

set(example_programs
    simple
    two_var
    ${openmp_programs}
   )

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)
  source_group("Source Files" FILES ${sources})
  add_executable("${example_program}_kokkos" ${sources})
  if (OPENMP_FOUND)
    set_target_properties("${example_program}_kokkos" PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
    target_link_libraries ("${example_program}_kokkos" apex ${Kokkos_LIBRARY} ${LIBS} OpenMP::OpenMP_CXX)
  else (OPENMP_FOUND)
    target_link_libraries ("${example_program}_kokkos" apex ${Kokkos_LIBRARY} ${LIBS})
  endif (OPENMP_FOUND)
  add_dependencies ("${example_program}_kokkos" apex)
  add_dependencies (tests "${example_program}_kokkos")
  add_test ("test_${example_program}_kokkos" "${example_program}_kokkos")
  set_tests_properties("test_${example_program}_kokkos" PROPERTIES TIMEOUT 30)
  set_property(TEST "test_${example_program}_kokkos" APPEND PROPERTY ENVIRONMENT "KOKKOS_TOOLS_LIBS=${PROJECT_BINARY_DIR}/src/apex/libapex${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set_property(TEST "test_${example_program}_kokkos" APPEND PROPERTY ENVIRONMENT "APEX_SCREEN_OUTPUT=1")
  set_property(TEST "test_${example_program}_kokkos" APPEND PROPERTY ENVIRONMENT "OMP_PROC_BIND=spread")
  set_property(TEST "test_${example_program}_kokkos" APPEND PROPERTY ENVIRONMENT "OMP_PLACES=threads")
  set_property(TEST "test_${example_program}_kokkos" APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")

endforeach()

