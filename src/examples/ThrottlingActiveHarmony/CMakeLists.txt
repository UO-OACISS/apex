# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testThrottlingActiveHarmony" that is built from the source file
# "testThrottlingActiveHarmony.c". The extensions are automatically found. 
add_executable (testThrottlingActiveHarmony testThrottlingActiveHarmony.c) 
add_dependencies (testThrottlingActiveHarmony apex)
add_dependencies (examples testThrottlingActiveHarmony)

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

# Link the executable to the Apex library. 
target_link_libraries (testThrottlingActiveHarmony apex) 

INSTALL(TARGETS testThrottlingActiveHarmony
  RUNTIME DESTINATION bin OPTIONAL
)
