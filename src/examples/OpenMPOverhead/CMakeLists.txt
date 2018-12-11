# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR}) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "openmp_overhead" that is built from the source file
# "openmp_overhead.cpp". The extensions are automatically found. 
add_executable (openmp_overhead openmp_overhead.c) 
add_dependencies (openmp_overhead apex)
add_dependencies (examples openmp_overhead)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Link the executable to the Apex library. 
target_link_libraries (openmp_overhead apex ${OMPT_LIBRARIES} ${LIBS} stdc++) 
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(openmp_overhead PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS openmp_overhead
  RUNTIME DESTINATION bin OPTIONAL
)
