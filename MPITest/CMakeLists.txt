# Make sure that spaces in linker lines don't cause CMake errors
if (POLICY CMP0004)
  cmake_policy(SET CMP0004 OLD)
endif()

# Make sure the compiler can find include files from our Apex library. 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}") 
include_directories (. ${APEX_SOURCE_DIR}/apex ${MPI_CXX_INCLUDE_PATH}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/apex) 

# Add executable called "mpi_test" that is built from the source file
# "mpi_test.cpp". The extensions are automatically found. 
add_executable (mpi_test mpi_test.cpp) 
#add_dependencies (mpi_test apexStatic apexShared)
add_dependencies (mpi_test apex)

# Link the executable to the Apex library. 
target_link_libraries (mpi_test apex stdc++ ${LIBS} ${Boost_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}) 

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS")

INSTALL(TARGETS mpi_test
  RUNTIME DESTINATION bin
)
