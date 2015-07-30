# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "lulesh_OpenMP_2.0" that is built from the source file
# "lulesh.cc". The extensions are automatically found. 
add_executable (lulesh_OpenMP_2.0 lulesh.cc lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc)
add_dependencies (lulesh_OpenMP_2.0 apex)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 -DUSE_MPI=0 -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -DUSE_MPI=0 -Wall")

# Link the executable to the Apex library. 
target_link_libraries (lulesh_OpenMP_2.0 apex ${LIBS} ${Boost_LIBRARIES} ${OMPT_LIBRARIES}) 

INSTALL(TARGETS lulesh_OpenMP_2.0
  RUNTIME DESTINATION bin
)
