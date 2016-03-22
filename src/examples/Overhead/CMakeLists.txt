# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testOverhead" that is built from the source file
# "testOverhead.cpp". The extensions are automatically found. 
add_executable (testOverhead testOverhead.cpp) 
add_dependencies (testOverhead apex)
add_dependencies (examples testOverhead)

# Link the executable to the Apex library. 
target_link_libraries (testOverhead apex ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testOverhead PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testOverhead
  RUNTIME DESTINATION bin OPTIONAL
)
