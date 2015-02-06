# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/TestThreads) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/TestThreads) 

# Add executable called "testThreads" that is built from the source file
# "testThreads.cpp". The extensions are automatically found. 
add_executable (testThreads testThreads.cpp) 
#add_dependencies (testThreads apexStatic apexShared)
add_dependencies (testThreads apex)

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testThreads apex ${Boost_LIBRARIES}) 

INSTALL(TARGETS testThreads
  RUNTIME DESTINATION bin
)
