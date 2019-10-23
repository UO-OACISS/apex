#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(ZLIB)

if (ZLIB_FOUND)

  # Add an imported target
  add_library(apex::zlib INTERFACE IMPORTED)
  set_property(TARGET apex::zlib PROPERTY
    INTERFACE_LINK_LIBRARIES ${ZLIB_LIBRARIES})
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ZLIB_LIBRARY_DIR})
  message(INFO " Using zlib: ${ZLIB_LIBRARY_DIR} ${ZLIB_LIBRARIES}")

  list(APPEND _apex_imported_targets apex::zlib)

endif()
