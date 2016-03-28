# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testPolicyEngine" that is built from the source file
# "testPolicyEngine.c". The extensions are automatically found. 
add_executable (testPolicyEngine testPolicyEngine.c) 
add_dependencies (testPolicyEngine apex)
add_dependencies (examples testPolicyEngine)

# Link the executable to the Apex library. 
target_link_libraries (testPolicyEngine apex ${LIBS} stdc++)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testPolicyEngine PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testPolicyEngine
  RUNTIME DESTINATION bin OPTIONAL
)
