# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/ITTNotify) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/ITTNotify) 

# Add executable called "testITT" that is built from the source file
# "testITT.cpp". The extensions are automatically found. 
add_executable (testITT testITT.cpp) 
target_link_libraries (testITT ittnotify apex ${LIBS}) 
add_dependencies (testITT ittnotify apex)
add_dependencies (examples testITT)

# Link the executable to the Apex library. 
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testITT PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testITT
  RUNTIME DESTINATION bin OPTIONAL
)
