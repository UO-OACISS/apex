# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/apex ${APEX_SOURCE_DIR}/PolicyEngineExample) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex ${APEX_BINARY_DIR}/PolicyEngineExample) 

# Add executable called "testPolicyEngine" that is built from the source file
# "testPolicyEngine.c". The extensions are automatically found. 
add_executable (testPolicyEngine testPolicyEngine.c) 
add_dependencies (testPolicyEngine apexStatic apexShared)

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (testPolicyEngine apex ${Boost_LIBRARIES}) 

INSTALL(TARGETS testPolicyEngine
  RUNTIME DESTINATION bin
)
