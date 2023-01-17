# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

set(util_programs
    apex_make_default_config
    apex_environment_help
   )

foreach(util_program ${util_programs})
  set(sources ${util_program}.cpp)
  source_group("Source Files" FILES ${sources})
  add_executable("${util_program}" ${sources})
  target_link_libraries ("${util_program}" apex stdc++ ${LIBS})
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties("${util_program}" PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
  add_dependencies ("${util_program}" apex)
  install(TARGETS "${util_program}" RUNTIME DESTINATION "bin" OPTIONAL)
endforeach()

