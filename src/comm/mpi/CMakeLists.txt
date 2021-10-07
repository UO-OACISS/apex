# Make sure that spaces in linker lines don't cause CMake errors
#if (POLICY CMP0004)
#  cmake_policy(SET CMP0004 OLD)
#endif()

message(INFO ${MPI_INCLUDE_PATH})
include_directories(${MPI_INCLUDE_PATH})

# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/comm ${APEX_SOURCE_DIR}/src/apex ${MPI_C_INCLUDE_PATH})
link_directories (${APEX_BINARY_DIR}/src/apex)

# Create a library called "apex_mpi" which includes the source file
# "apex_global_mpi.c".
# The extension is already found. Any number of sources could be listed here.
add_library (apex_mpi apex_global_mpi.c)
set_target_properties (apex_mpi PROPERTIES OUTPUT_NAME apex_mpi)

if(MPI_COMPILE_FLAGS)
    set_target_properties (apex_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif(MPI_COMPILE_FLAGS)

if(MPI_LINK_FLAGS)
    set_target_properties (apex_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif(MPI_LINK_FLAGS)

target_link_libraries (apex_mpi apex ${MPI_LIBRARIES})

if(OpenMP_CXX_FOUND)
set_target_properties (apex_mpi PROPERTIES LINK_FLAGS "${OpenMP_C_FLAGS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

# add the binary tree to the search path for include files
# so that we will find ittnotify.h
include_directories ("${PROJECT_BINARY_DIR}/src/apex" ${APEX_BINARY_DIR}/src/comm)

INSTALL(TARGETS apex_mpi
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

