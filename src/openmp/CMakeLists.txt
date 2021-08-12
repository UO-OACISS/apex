# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if (OpenMP_FOUND) # don't think we'd be in here otherwise...
    if (CUPTI_FOUND)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
            set(OMPT_TARGET_TEST ompt_target ompt_target_matmult ompt_target_daxpy ompt_target_vector_add)
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(omp_target_flags "${OpenMP_C_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda")
            check_cxx_compiler_flag("${omp_target_flags}" flag_supported)
            if(flag_supported)
                message(INFO " ${omp_target_flags} supported by compiler, enabling it")
                set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${omp_target_flags}")
                set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${omp_target_flags}")
                set(OMPT_TARGET_TEST ompt_target ompt_target_matmult ompt_target_daxpy ompt_target_vector_add)
            else()
                set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
                set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            endif()
            unset(flag_supported CACHE)
        endif()
    endif (CUPTI_FOUND)
    if (ROCTRACER_FOUND)
        set (omp_target_flags "${OpenMP_C_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906")
        check_cxx_compiler_flag("${omp_target_flags}" flag_supported)
        if(flag_supported)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${omp_target_flags")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${omp_target_flags")
            set(OMPT_TARGET_TEST ompt_target ompt_target_matmult ompt_target_daxpy ompt_target_vector_add)
        else()
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif (ROCTRACER_FOUND)
endif (OpenMP_FOUND)

set(example_programs
    ompt_thread
    ompt_parallel_region
    ompt_reduction
    ompt_task
    ompt_single
    ompt_sections
    ompt_sync_region_wait
    ompt_master
    ${OMPT_TARGET_TEST}
   )

foreach(example_program ${example_programs})
    set(sources ${example_program}.c)

    source_group("Source Files" FILES ${sources})

    add_executable(${example_program} ${sources})
    target_link_libraries (${example_program} ${OMPT_LIBRARIES} apex ${LIBS} stdc++ m)
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties(${example_program} PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
    add_dependencies (${example_program} apex)
    add_dependencies (tests ${example_program})
    add_test ("test_${example_program}" ${example_program})
    set_tests_properties("test_${example_program}" PROPERTIES ENVIRONMENT "APEX_SCREEN_OUTPUT=1;APEX_OMPT_HIGH_OVERHEAD_EVENTS=1")
endforeach()

