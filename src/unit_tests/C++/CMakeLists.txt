# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

if(OpenMP_FOUND)
    set(APEX_OPENMP_TEST apex_openmp)
endif(OpenMP_FOUND)

set(example_programs
    apex_init
    apex_init_args
    apex_finalize
    apex_cleanup
    apex_exit
    apex_start
    apex_stop
    apex_yield
    apex_resume
    apex_reset
    apex_dump
    apex_set_state
    apex_sample_value
    apex_register_custom_event
    apex_custom_event
    apex_version
    apex_register_thread
    apex_exit_thread
    apex_register_policy
    apex_register_policy_set
    apex_register_periodic_policy
    apex_stop_all_async_threads
    apex_deregister_policy
    apex_get_profile
    apex_current_power_high
    apex_setup_timer_throttling
    apex_print_options
    apex_get_thread_cap
    apex_shutdown_throttling
    apex_hpx_direct_actions
    apex_hpx_task_wrapper_direct_actions
    apex_hpx_annotated_functions
    apex_profiler_guids
    apex_non_worker_thread
    apex_swap_threads
    apex_malloc
    apex_std_thread
    ${APEX_OPENMP_TEST}
   )
    #apex_set_thread_cap
    #apex_setup_power_cap_throttling
    #apex_setup_custom_tuning
    #get_tunable_params

if (OPENMP_FOUND)
  set(example_programs "${example_programs};apex_setup_throughput_tuning")
endif (OPENMP_FOUND)

if ((NOT PAPI_FOUND) AND (NOT OTF2_FOUND))
    set(example_programs "${example_programs};apex_pthread_flood")
endif()

# std::threads crash when linked statically. :(
if (NOT BUILD_STATIC_EXECUTABLES)
# Intel can't do std::futures (std::__once_callable)
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  #if ((NOT OTF2_FOUND))
    set(example_programs "${example_programs};apex_fibonacci_std_async;apex_fibonacci_std_async2")
  #endif()
  set(example_programs "${example_programs};apex_new_task;apex_task_wrapper;apex_task_wrapper2")
  if ((NOT DEFINED TAU_ROOT) AND (NOT APEX_WITH_TAU) AND (NOT TAU_FOUND))
    if (APEX_THROTTLE)
      set(example_programs "${example_programs};apex_throttle_event")
    endif (APEX_THROTTLE)
  endif ((NOT DEFINED TAU_ROOT) AND (NOT APEX_WITH_TAU) AND (NOT TAU_FOUND))
endif (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif()

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)
  source_group("Source Files" FILES ${sources})
  add_executable("${example_program}_cpp" ${sources})
  target_link_libraries ("${example_program}_cpp" apex ${LIBS})
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties("${example_program}_cpp" PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
  add_dependencies ("${example_program}_cpp" apex)
  add_dependencies (tests "${example_program}_cpp")
  add_test ("test_${example_program}_cpp" "${example_program}_cpp")
  set_tests_properties("test_${example_program}_cpp" PROPERTIES TIMEOUT 30)
  # install(TARGETS "${example_program}_cpp" RUNTIME DESTINATION "bin/apex_unit_tests" OPTIONAL)
endforeach()
if (OPENMP_FOUND)
    set_tests_properties("test_apex_setup_throughput_tuning_cpp" PROPERTIES TIMEOUT 120)
endif (OPENMP_FOUND)

if (OPENMP_FOUND)
  set_target_properties(apex_setup_throughput_tuning_cpp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(apex_setup_throughput_tuning_cpp PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  if (PROCESSOR_COUNT)
    set_property (TEST test_apex_setup_throughput_tuning_cpp
    APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
    set_property (TEST test_apex_setup_throughput_tuning_cpp
        APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=TRUE")
  endif (PROCESSOR_COUNT)
  set_target_properties(apex_openmp_cpp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(apex_openmp_cpp PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif (OPENMP_FOUND)

#set_target_properties(apex_fibonacci_std_async_cpp PROPERTIES LINK_FLAGS "-fsplit-stack")
#target_link_libraries (apex_fibonacci_std_async_cpp apex_pthread_wrapper)
#add_dependencies (apex_fibonacci_std_async_cpp apex_pthread_wrapper)


set_property (TEST test_apex_malloc_cpp APPEND PROPERTY ENVIRONMENT
    "LD_PRELOAD=${APEX_BINARY_DIR}/src/wrappers/libapex_memory_wrapper${CMAKE_SHARED_LIBRARY_SUFFIX}")
set_property (TEST test_apex_malloc_cpp APPEND PROPERTY ENVIRONMENT
    "APEX_PROC_STAT=0")
set_property (TEST test_apex_malloc_cpp APPEND PROPERTY ENVIRONMENT
    "APEX_TRACK_CPU_MEMORY=1")

set_tests_properties(test_apex_version_cpp PROPERTIES ENVIRONMENT "APEX_PROC_SELF_STATUS=0;APEX_PROC_STAT=0")

# Make sure the compiler can find include files from our Apex library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${MPI_CXX_INCLUDE_PATH})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "apex_hpx_annotated_functions_mpi" that is built from the source file
# "apex_hpx_annotated_functions.cpp". The extensions are automatically found.
add_executable (apex_hpx_annotated_functions_mpi apex_hpx_annotated_functions.cpp)
add_dependencies (apex_hpx_annotated_functions_mpi apex)
add_dependencies (examples apex_hpx_annotated_functions_mpi)

# Link the executable to the Apex library.
target_link_libraries (apex_hpx_annotated_functions_mpi apex ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS} ${APEX_STDCXX_LIB} m)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(apex_hpx_annotated_functions_mpi PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS apex_hpx_annotated_functions_mpi
  RUNTIME DESTINATION bin OPTIONAL
)

