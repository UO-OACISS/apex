# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "openmp_test" that is built from the source file
# "openmp_test.cpp". The extensions are automatically found. 
add_executable (openmp_test openmp_test.c) 
add_dependencies (openmp_test apex)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# Link the executable to the Apex library. 
target_link_libraries (openmp_test apex ${Boost_LIBRARIES} ${OMPT_LIBRARIES}) 

INSTALL(TARGETS openmp_test
  RUNTIME DESTINATION bin
)