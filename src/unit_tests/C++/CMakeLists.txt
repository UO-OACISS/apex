# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper) 

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
    apex_set_state
    apex_sample_value
    apex_register_custom_event
    apex_custom_event
    apex_version
    apex_set_node_id
    apex_register_thread
    apex_exit_thread
    apex_register_policy
    apex_register_policy_set
    apex_register_periodic_policy
    apex_deregister_policy
    apex_get_profile
    apex_current_power_high
    apex_setup_timer_throttling
    apex_print_options
    apex_get_thread_cap
    apex_shutdown_throttling
   )
    #apex_fibonacci_std_async
    #apex_set_thread_cap
    #apex_setup_power_cap_throttling
    #apex_setup_custom_tuning
    #get_tunable_params

if (OPENMP_FOUND)
  set(example_programs "${example_programs};apex_setup_throughput_tuning")
endif (OPENMP_FOUND)

if (NOT PAPI_FOUND)
    set(example_programs "${example_programs};apex_pthread_flood")
endif()

# std::threads crash when linked statically. :(
if (NOT BUILD_STATIC_EXECUTABLES)
  set(example_programs "${example_programs};apex_new_task")
  if ((NOT DEFINED TAU_ROOT) AND (NOT USE_TAU) AND (NOT TAU_FOUND) AND (NOT USE_SOS))
    set(example_programs "${example_programs};apex_throttle_event")
  endif ((NOT DEFINED TAU_ROOT) AND (NOT USE_TAU) AND (NOT TAU_FOUND) AND (NOT USE_SOS))
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
  # install(TARGETS "${example_program}_cpp" RUNTIME DESTINATION "bin/apex_unit_tests" OPTIONAL)
endforeach()

if (OPENMP_FOUND)
  set_target_properties(apex_setup_throughput_tuning_cpp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(apex_setup_throughput_tuning_cpp PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  if (PROCESSOR_COUNT)
    set_property (TEST test_apex_setup_throughput_tuning_cpp
    APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
    set_property (TEST test_apex_setup_throughput_tuning_cpp
        APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=TRUE")
  endif (PROCESSOR_COUNT)
endif (OPENMP_FOUND)

#set_target_properties(apex_fibonacci_std_async_cpp PROPERTIES LINK_FLAGS "-fsplit-stack")
#target_link_libraries (apex_fibonacci_std_async_cpp apex_pthread_wrapper)        
#add_dependencies (apex_fibonacci_std_async_cpp apex_pthread_wrapper)    

