#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(BFD)

if (BFD_FOUND)

  add_library(apex::bfd INTERFACE IMPORTED)
  set_property(TARGET apex::bfd PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${BFD_INCLUDE_DIRS})
  set_property(TARGET apex::bfd PROPERTY
    INTERFACE_LINK_LIBRARIES ${BFD_LIBRARIES})
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${BFD_LIBRARY_DIR})
  message(INFO " Using binutils: ${BFD_LIBRARY_DIR} ${BFD_LIBRARIES}")

  list(APPEND _apex_imported_targets apex::bfd)

endif()
