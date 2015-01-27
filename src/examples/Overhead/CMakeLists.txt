# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/Overhead) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/Overhead) 

# Add executable called "testOverhead" that is built from the source file
# "testOverhead.cpp". The extensions are automatically found. 
add_executable (testOverhead testOverhead.cpp) 
#add_dependencies (testOverhead apexStatic apexShared)
add_dependencies (testOverhead apex)

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (testOverhead apex ${Boost_LIBRARIES} ${LIBS}) 

INSTALL(TARGETS testOverhead
  RUNTIME DESTINATION bin
)
