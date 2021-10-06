# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "openmp_test" that is built from the source file
# "openmp_test.cpp". The extensions are automatically found.
add_executable (openmp_test openmp_test.c)
add_dependencies (openmp_test apex)
add_dependencies (examples openmp_test)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# Link the executable to the Apex library.
target_link_libraries (openmp_test apex ${LIBS} ${OMPT_LIBRARIES} stdc++)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(openmp_test PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS openmp_test
  RUNTIME DESTINATION bin OPTIONAL
)
