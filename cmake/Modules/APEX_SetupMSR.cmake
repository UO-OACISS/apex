#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_MSR)

  find_package(MSR)
  if(NOT MSR_FOUND)
      hpx_error("apex" "Requested APEX_WITH_MSR but could not find MSR. \
        Please specify MSR_ROOT.")
  endif()
  hpx_info("apex" "Building APEX with libmsr support.")

  # Add an imported target
  add_library(msr INTERFACE IMPORTED)
  set_property(TARGET msr PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${MSR_INCLUDE_DIR})
  set_property(TARGET msr PROPERTY
    INTERFACE_LINK_LIBRARIES ${MSR_LIBRARIES})

  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${MSR_LIBRARY_DIR})
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_MSR)

  list(APPEND _apex_imported_targets msr)

endif()
