# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "count_calls" that is built from the source file
# "count_calls.cpp". The extensions are automatically found. 
add_executable (count_calls count_calls.cpp) 
add_dependencies (count_calls apex)
add_dependencies (examples count_calls)

# Link the executable to the Apex library. 
target_link_libraries (count_calls apex ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(count_calls PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS count_calls
  RUNTIME DESTINATION bin OPTIONAL
)
