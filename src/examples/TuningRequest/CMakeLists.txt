# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 

# Add executable called "tuning_request" that is built from the source file
# "tuning_request.cpp". The extensions are automatically found. 
add_executable (tuning_request tuning_request.cpp) 
add_dependencies (tuning_request apex)
add_dependencies (examples tuning_request)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Link the executable to the Apex library. 
target_link_libraries (tuning_request apex ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(tuning_request PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS tuning_request
  RUNTIME DESTINATION bin OPTIONAL
)
