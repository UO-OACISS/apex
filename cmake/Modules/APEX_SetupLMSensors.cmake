#  Copyright (c) 2014 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# LM Sensors configuration
################################################################################

# Setup an imported target for lmsensors
if(APEX_WITH_LM_SENSORS)

  find_package(LM_SENSORS)

  if (NOT LM_SENSORS_FOUND)
    hpx_error("apex" "Requested APEX_WITH_LM_SENSORS but could not find LM \
    Sensors. Please specify LM_SENSORS_ROOT.")
  endif()

  # Add an imported target
  add_library(lm_sensors INTERFACE IMPORTED)
  set_property(TARGET lm_sensors PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${LM_SENSORS_INCLUDE_DIRS})
  set_property(TARGET lm_sensors PROPERTY
    INTERFACE_LINK_LIBRARIES ${LM_SENSORS_LIBRARIES})
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${LM_SENSORS_LIBRARY_DIR})

  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_LM_SENSORS)

  list(APPEND _apex_imported_targets lm_sensors)

endif()

