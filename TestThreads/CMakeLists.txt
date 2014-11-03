# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/apex ${APEX_SOURCE_DIR}/TestThreads) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex ${APEX_BINARY_DIR}/TestThreads) 

# Add executable called "testThreads" that is built from the source file
# "testThreads.cpp". The extensions are automatically found. 
add_executable (testThreads testThreads.cpp) 
#add_dependencies (testThreads apexStatic apexShared)
add_dependencies (testThreads apex)

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (testThreads apex ${Boost_LIBRARIES}) 

INSTALL(TARGETS testThreads
  RUNTIME DESTINATION bin
)
