# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testBeacon" that is built from the source file
# "testBeacon.cpp". The extensions are automatically found. 
add_executable (testBeacon testBeacon.cpp) 
add_dependencies (testBeacon apex)
add_dependencies (examples testBeacon)

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testBeacon apex)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_script.in ${CMAKE_CURRENT_BINARY_DIR}/test_script.sh)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/test_script.sh
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
      GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

INSTALL(TARGETS testBeacon
  RUNTIME DESTINATION bin OPTIONAL
)
