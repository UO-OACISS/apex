# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "apexCppDemo" that is built from the source file
# "demo.cpp". The extensions are automatically found.
add_executable (eventFilterExample eventfilter_example.cpp)

# Link the executable to the Apex library.
target_link_libraries (eventFilterExample apex ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(eventFilterExample PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()
add_dependencies (eventFilterExample apex)
add_dependencies (examples eventFilterExample)

INSTALL(TARGETS eventFilterExample
  RUNTIME DESTINATION bin OPTIONAL
)
