# Make sure the compiler can find include files from our Apex library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${MPI_CXX_INCLUDE_PATH})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "mpi_test" that is built from the source file
# "mpi_cpi.c". The extensions are automatically found.
add_executable (mpi_cpi mpi_cpi.c)
add_dependencies (mpi_cpi apex)
add_executable (mpi_divergent mpi_divergent.c)
add_dependencies (mpi_divergent apex)
add_dependencies (tests mpi_cpi mpi_divergent)

# Link the executable to the Apex library.
target_link_libraries (mpi_cpi apex apex_mpi ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS} ${APEX_STDCXX_LIB} m)
target_link_libraries (mpi_divergent apex apex_mpi ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS} ${APEX_STDCXX_LIB} m)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(mpi_cpi PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    set_target_properties(mpi_divergent PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS mpi_cpi mpi_divergent
  RUNTIME DESTINATION bin OPTIONAL
)
