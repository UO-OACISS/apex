# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testThrottling" that is built from the source file
# "testThrottling.c". The extensions are automatically found. 
add_executable (testThrottling testThrottling.c) 
add_dependencies (testThrottling apex)
add_dependencies (examples testThrottling)

# Link the executable to the Apex library. 
target_link_libraries (testThrottling apex stdc++ ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testThrottling PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testThrottling
  RUNTIME DESTINATION bin OPTIONAL
)
