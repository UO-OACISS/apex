# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "policyUnitTest" that is built from the source file
# "policyUnitTest.c". The extensions are automatically found. 
add_executable (policyUnitTest policyUnitTest.c) 
add_dependencies (policyUnitTest apex)
add_dependencies (examples policyUnitTest)

# Link the executable to the Apex library. 
target_link_libraries (policyUnitTest apex ${LIBS} stdc++ m)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(policyUnitTest PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS policyUnitTest
  RUNTIME DESTINATION bin OPTIONAL
)
