# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex) 

# Add executable called "apexCDemo" that is built from the source file
# "demo.c". The extensions are automatically found. 
add_executable (apexCDemo demo.c) 

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (apexCDemo apex stdc++) 
#add_dependencies (apexCDemo apexStatic apexShared)
add_dependencies (apexCDemo apex)

INSTALL(TARGETS apexCDemo
  RUNTIME DESTINATION bin
)
