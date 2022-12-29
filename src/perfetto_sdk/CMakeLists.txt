include_directories(${APEX_SOURCE_DIR}/src/perfetto_sdk)
add_library (perfetto perfetto.cc)

if (WIN32)
  # The perfetto library contains many symbols, so it needs the big object
  # format.
  target_compile_options(perfetto PRIVATE "/bigobj")
  # Disable legacy features in windows.h.
  add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
  # On Windows we should link to WinSock2.
  add_dependencies (perfetto ws2_32)
endif (WIN32)

# Enable standards-compliant mode when using the Visual Studio compiler.
if (MSVC)
  target_compile_options(example PRIVATE "/permissive-")
endif (MSVC)

INSTALL(TARGETS perfetto
    PUBLIC_HEADER DESTINATION include
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
