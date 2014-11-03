# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/apex ${APEX_SOURCE_DIR}/PolicyEngineCppExample) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex ${APEX_BINARY_DIR}/PolicyEngineCppExample) 

# Add executable called "testPolicyEngine" that is built from the source file
# "testPolicyEngine.cpp". The extensions are automatically found. 
add_executable (testCppPolicyEngine testPolicyEngine.cpp) 
#add_dependencies (testCppPolicyEngine apexStatic apexShared)
add_dependencies (testCppPolicyEngine apex)

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (testCppPolicyEngine apex ${Boost_LIBRARIES} ${LIBS}) 

INSTALL(TARGETS testCppPolicyEngine
  RUNTIME DESTINATION bin
)
