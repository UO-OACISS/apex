# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "1d_stencil" that is built from the source file
# "1d_stencil.cpp". The extensions are automatically found. 
add_executable (1d_stencil 1d_stencil.cpp) 
add_dependencies (1d_stencil apex)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Link the executable to the Apex library. 
target_link_libraries (1d_stencil apex ${Boost_LIBRARIES} ${OMPT_LIBRARIES}) 

INSTALL(TARGETS 1d_stencil
  RUNTIME DESTINATION bin
)
