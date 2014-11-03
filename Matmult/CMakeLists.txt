# Make sure the compiler can find include files from our Apex library. 
include_directories (. ${APEX_SOURCE_DIR}/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex) 

# Add executable called "apexCDemo" that is built from the source file
# "demo.c". The extensions are automatically found. 
add_executable (matmult matmult_initialize.cpp matmult.cpp) 
#add_dependencies (matmult apexStatic apexShared)
add_dependencies (matmult apex)

# Link the executable to the Apex library. 
target_link_libraries (matmult apex stdc++ ${LIBS} ${Boost_LIBRARIES}) 

## Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")

INSTALL(TARGETS matmult
  RUNTIME DESTINATION bin
)
