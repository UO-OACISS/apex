# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add executable called "testThrottlingActiveHarmony" that is built from the source file
# "testThrottlingActiveHarmony.c". The extensions are automatically found.
add_executable (testThrottlingActiveHarmony testThrottlingActiveHarmony.c)
add_dependencies (testThrottlingActiveHarmony apex)
add_dependencies (examples testThrottlingActiveHarmony)

# Link the executable to the Apex library.
target_link_libraries (testThrottlingActiveHarmony apex ${LIBS} ${APEX_STDCXX_LIB} m)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(testThrottlingActiveHarmony PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS testThrottlingActiveHarmony
  RUNTIME DESTINATION bin OPTIONAL
)
