#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_OTF2)

  find_package(OTF2 REQUIRED)
  if (NOT OTF2_FOUND)
      hpx_error("apex" "Requested APEX_WITH_OTF2 but could not find OTF2 library. Please specify OTF2_ROOT.")
  endif()

  # Add an imported target
  add_library(otf2 INTERFACE IMPORTED)
  set_property(TARGET otf2 PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${OTF2_INCLUDE_DIRS})
  set_property(TARGET otf2 PROPERTY
    INTERFACE_LINK_LIBRARIES ${OTF2_LIBRARIES})

  set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OTF2_LIBRARY_DIR})

  list(APPEND _apex_imported_targets otf2)

else()
  add_custom_target(project_otf2)
endif()

