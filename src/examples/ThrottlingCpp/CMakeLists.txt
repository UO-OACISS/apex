# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testThrottlingCpp" that is built from the source file
# "testThrottling.cpp". The extensions are automatically found. 
add_executable (testThrottlingCpp testThrottling.cpp) 
add_dependencies (testThrottlingCpp apex)

if(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testThrottlingCpp apex ${Boost_LIBRARIES} ${LIBS}) 

INSTALL(TARGETS testThrottlingCpp
  RUNTIME DESTINATION bin
)
