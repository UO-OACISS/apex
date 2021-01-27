#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_ACTIVEHARMONY)

  find_package(ACTIVEHARMONY)
  if(NOT ACTIVEHARMONY_FOUND)
      hpx_error("apex" "Requested APEX_WITH_ACTIVEHARMONY but could not find \
      Active Harmony. Please specify ACTIVEHARMONY_ROOT.")
  endif()

  # Add an imported target
  add_library(activeharmony INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with Active Harmony support.")
  set_property(TARGET activeharmony PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${ACTIVEHARMONY_INCLUDE_DIR})
  set_property(TARGET activeharmony PROPERTY
    INTERFACE_LINK_LIBRARIES ${ACTIVEHARMONY_LIBRARIES})
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ACTIVEHARMONY_LIBRARY_DIR})

  # Add the right definitions to the apex_flags target
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_ACTIVEHARMONY)

  list(APPEND _apex_imported_targets activeharmony)

else()

  add_custom_target(project_activeharmony)

endif()
