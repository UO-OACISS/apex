# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "testThreads" that is built from the source file
# "testThreads.cpp". The extensions are automatically found. 
add_executable (testThreads testThreads.cpp) 
add_dependencies (testThreads apex)
add_dependencies (examples testThreads)

# Link the executable to the Apex library. 
target_link_libraries (testThreads apex ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testThreads PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testThreads
  RUNTIME DESTINATION bin OPTIONAL
)
