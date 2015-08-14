# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "testpower" that is built from the source file
# "getpower.cpp". The extensions are automatically found. 
add_executable (testpower getpower.cpp) 

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testpower apex)
add_dependencies (testpower apex)

INSTALL(TARGETS testpower
  RUNTIME DESTINATION bin
)
