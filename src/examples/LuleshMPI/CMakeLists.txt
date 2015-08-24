# Make sure that spaces in linker lines don't cause CMake errors
if (POLICY CMP0004)
  cmake_policy(SET CMP0004 OLD)
endif()

# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/comm ${MPI_CXX_INCLUDE_PATH}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/comm/mpi) 

# Add executable called "lulesh_MPI_2.0" that is built from the source file
# "lulesh.cc". The extensions are automatically found. 
add_executable (lulesh_MPI_2.0 lulesh.cc lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc)
add_dependencies (lulesh_MPI_2.0 apex)
add_dependencies (examples lulesh_MPI_2.0)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DUSE_MPI=1")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DUSE_MPI=1")

# Link the executable to the Apex library. 
target_link_libraries (lulesh_MPI_2.0 apex apex_mpi stdc++)
${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})

INSTALL(TARGETS lulesh_MPI_2.0
  RUNTIME DESTINATION bin OPTIONAL
)
