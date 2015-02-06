# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "testReset" that is built from the source file
# "reset.cpp". The extensions are automatically found. 
add_executable (testReset reset.cpp) 

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testReset apex ${Boost_LIBRARIES})
#add_dependencies (testReset apexStatic apexShared)
add_dependencies (testReset apex)

INSTALL(TARGETS testReset
  RUNTIME DESTINATION bin
)
