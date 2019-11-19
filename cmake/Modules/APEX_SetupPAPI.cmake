#  Copyright (c) 2014 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Setup an imported target for papi
if(APEX_WITH_PAPI)

  find_package(PAPI)
  if(NOT PAPI_FOUND)
      hpx_error("apex" "Requested APEX_WITH_PAPI but could not find PAPI. Please specify PAPI_ROOT.")
  endif()
  hpx_info("apex" "Building APEX with PAPI support.")

  # Add an imported target
  add_library(papi INTERFACE IMPORTED)
  set_property(TARGET papi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${PAPI_INCLUDE_DIR})
  set_property(TARGET papi PROPERTY
    INTERFACE_LINK_LIBRARIES ${PAPI_LIBRARIES})

  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${PAPI_LIBRARY_DIR})
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_PAPI)

  list(APPEND _apex_imported_targets papi)

endif()
