# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/Apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/Apex) 

# Add executable called "apexCDemo" that is built from the source file
# "demo.c". The extensions are automatically found. 
add_executable (apexCDemo demo.c) 

# Link the executable to the Apex library. 
target_link_libraries (apexCDemo Apex stdc++) 
add_dependencies (apexCDemo ApexStatic ApexShared)

INSTALL(TARGETS apexCDemo
  RUNTIME DESTINATION bin
)
