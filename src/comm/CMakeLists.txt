set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}") 

# install the header.
INSTALL(FILES apex_global.h DESTINATION include)

