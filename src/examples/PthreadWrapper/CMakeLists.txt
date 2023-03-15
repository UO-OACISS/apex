# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

set(sources apex_pthread_wrap.c)
source_group("Source Files" FILES ${sources})

# add example executable
add_executable(pthread_wrapper_example ${sources})
target_link_libraries (pthread_wrapper_example apex apex_pthread_wrapper ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(pthread_wrapper_example PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()
add_dependencies (pthread_wrapper_example apex apex_pthread_wrapper)
add_test ("test_pthread_wrapper_example" pthread_wrapper_example)
set_tests_properties(pthread_wrapper_example PROPERTIES TIMEOUT 30)

