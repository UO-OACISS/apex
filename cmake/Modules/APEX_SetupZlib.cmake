#  Copyright (c) 2021 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Setup an imported target for zlib
find_package(ZLIB)
if(ZLIB_FOUND)
  hpx_info("apex" "Building APEX with ZLIB support.")

  # Add an imported target
  add_library(zlib INTERFACE IMPORTED)
  set_property(TARGET zlib PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIR})
  set_property(TARGET zlib PROPERTY
    INTERFACE_LINK_LIBRARIES ${ZLIB_LIBRARIES})

  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ZLIB_LIBRARY_DIR})
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_ZLIB)
  message(INFO " Using zlib: ${ZLIB_INCLUDE_DIR}")
  message(INFO " Using zlib: ${ZLIB_LIBRARY_DIR} ${ZLIB_LIBRARIES}")

  list(APPEND _apex_imported_targets zlib)

endif()
