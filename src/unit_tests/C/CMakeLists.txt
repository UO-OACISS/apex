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
    apex_register_periodic_policy
    apex_deregister_policy
    apex_get_profile
    apex_get_idle_rate
    apex_current_power_high
    apex_setup_timer_throttling
    apex_print_options
    apex_get_thread_cap
    apex_shutdown_throttling
    apex_disable
    apex_suspend
    apex_papi_suspend
    apex_cross_thread_timing
   )
#    apex_setup_power_cap_throttling
#    apex_setup_throughput_tuning
#    apex_set_thread_cap

if (NOT USE_TAU)
  set(example_programs "${example_programs};apex_fibonacci_pthread")
  set(example_programs "${example_programs};apex_new_task")
endif (NOT USE_TAU)

#if (OPENMP_FOUND)
  #set(example_programs "${example_programs} apex_setup_throughput_tuning")
#endif (OPENMP_FOUND)

foreach(example_program ${example_programs})
    set(sources ${example_program}.c)

    source_group("Source Files" FILES ${sources})

    add_executable(${example_program} ${sources})
    target_link_libraries (${example_program} apex stdc++ ${LIBS})
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties(${example_program} PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
    add_dependencies (${example_program} apex)
    add_dependencies (tests ${example_program})
    add_test ("test_${example_program}" ${example_program})
    # install(TARGETS ${example_program} RUNTIME DESTINATION "bin/apex_unit_tests" OPTIONAL)
endforeach()

set_tests_properties(test_apex_disable PROPERTIES ENVIRONMENT "APEX_DISABLE=1")
set_tests_properties(test_apex_disable PROPERTIES ENVIRONMENT "APEX_PAPI_METRICS=\"PAPI_TOT_CYC PAPI_TOT_INS\"")


