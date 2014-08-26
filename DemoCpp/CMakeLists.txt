# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex) 

# Add executable called "apexCppDemo" that is built from the source file
# "demo.cpp". The extensions are automatically found. 
add_executable (apexCppDemo demo.cpp) 

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

# Link the executable to the Apex library. 
target_link_libraries (apexCppDemo apex ${Boost_LIBRARIES}) 
add_dependencies (apexCppDemo apexStatic apexShared)

INSTALL(TARGETS apexCppDemo
  RUNTIME DESTINATION bin
)
