# Make sure that spaces in linker lines don't cause CMake errors
if (POLICY CMP0004)
  cmake_policy(SET CMP0004 OLD)
endif()

# Make sure the compiler can find include files from our Apex library. 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}") 
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/comm ${MPI_CXX_INCLUDE_PATH}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/comm/mpi) 

# Add executable called "mpi_imbalance_test" that is built from the source file
# "mpi_test.cpp". The extensions are automatically found. 
add_executable (mpi_imbalance_test mpi_test.cpp) 
add_dependencies (mpi_imbalance_test apex apex_mpi)

# Link the executable to the Apex library. 
target_link_libraries (mpi_imbalance_test apex apex_mpi stdc++ ${LIBS} ${Boost_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}) 

## Compiler flags
if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

INSTALL(TARGETS mpi_imbalance_test
  RUNTIME DESTINATION bin
)