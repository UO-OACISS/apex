# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex
${APEX_SOURCE_DIR}/src/wrappers)
# add the binary tree to the search path for include files
# so that we will find ApexConfig.h
include_directories("${PROJECT_BINARY_DIR}/src/apex")

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add library called "apex_pthread_wrapper" that is built from the source file
add_library (apex_pthread_wrapper pthread_wrapper.c pthread_wrapper_internal.cpp)
add_dependencies (apex_pthread_wrapper apex)
target_link_libraries (apex_pthread_wrapper apex)

# Add library called "apex_memory_wrapper" that is built from the source file
add_library (apex_memory_wrapper memory_wrapper.cpp memory_wrapper_internal.cpp)
add_dependencies (apex_memory_wrapper apex)
target_link_libraries (apex_memory_wrapper apex)

if(NOT APPLE)
    # Add library called "apex_dl_auditor" that is built from the source file
    add_library (apex_dl_auditor dl_auditor.c)
    target_link_libraries (apex_dl_auditor dl)
    INSTALL(TARGETS apex_pthread_wrapper apex_memory_wrapper apex_dl_auditor
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else(NOT APPLE)
    INSTALL(TARGETS apex_pthread_wrapper apex_memory_wrapper
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif(NOT APPLE)
