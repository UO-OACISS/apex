#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(Demangle)

if (DEMANGLE_FOUND)

  # Add an imported target
  add_library(apex::demangle INTERFACE IMPORTED)
  set_property(TARGET apex::demangle PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${DEMANGLE_INCLUDE_DIRS})
  set_property(TARGET apex::demangle PROPERTY
    INTERFACE_LINK_LIBRARIES ${DEMANGLE_LIBRARIES})

	set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${DEMANGLE_LIBRARY_DIR})
  message(INFO " Using demangle: ${DEMANGLE_LIBRARY_DIR} ${DEMANGLE_LIBRARIES}")

  list(APPEND _apex_imported_targets apex::demangle)

else()

	unset(DEMANGLE_LIBRARY)
	unset(DEMANGLE_LIBRARIES)
	unset(DEMANGLE_DIR)

endif()
