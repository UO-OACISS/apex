# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

set(example_programs
    apex_init
    apex_init_args
    apex_finalize
    apex_cleanup
	apex_start
	apex_stop
	apex_yield
	apex_resume
	apex_reset
   )

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.c)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_executable(${example_program} ${sources})
  target_link_libraries (${example_program} apex)
  add_dependencies (${example_program} apex)
  add_test ("test_${example_program}" ${example_program})
endforeach()

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

