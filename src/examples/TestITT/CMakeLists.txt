# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/ITTNotify) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/ITTNotify) 

# Add executable called "testITT" that is built from the source file
# "testITT.cpp". The extensions are automatically found. 
add_executable (testITT testITT.cpp) 
add_dependencies (testITT apex ittnotify)

# Link the executable to the Apex library. 
target_link_libraries (testITT apex ittnotify ${Boost_LIBRARIES}) 

INSTALL(TARGETS testITT
  RUNTIME DESTINATION bin
)
