#  Copyright (c) 2020-2021 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_HIP)
  find_package(ROCTRACER REQUIRED QUIET COMPONENTS ROCTRACER)
  find_package(ROCPROFILER REQUIRED QUIET COMPONENTS ROCPROFILER)
  find_package(ROCTX REQUIRED QUIET COMPONENTS ROCTX)
  find_package(RSMI REQUIRED)

  # Add an imported target
  add_library(roctracer INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with HIP/ROCTRACER support.")
  add_library(rocprofiler INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with HIP/ROCPROFILER support.")
  add_library(roctx INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with HIP/ROCTX support.")
  add_library(rsmi INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with HIP/ROC-SMI support.")

  set_property(TARGET roctracer PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${ROCTRACER_INCLUDE_DIRS})
  set_property(TARGET roctracer PROPERTY
    INTERFACE_LINK_LIBRARIES ${ROCTRACER_LIBRARIES})
  set_property(TARGET rocprofiler PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${ROCPROFILER_INCLUDE_DIRS})
  set_property(TARGET rocprofiler PROPERTY
    INTERFACE_LINK_LIBRARIES ${ROCPROFILER_LIBRARIES})
  set_property(TARGET roctx PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${ROCTX_INCLUDE_DIRS})
  set_property(TARGET roctx PROPERTY
    INTERFACE_LINK_LIBRARIES ${ROCTX_LIBRARIES})
  set_property(TARGET rsmi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${RSMI_INCLUDE_DIRS})
  set_property(TARGET rsmi PROPERTY
    INTERFACE_LINK_LIBRARIES ${RSMI_LIBRARIES})

  # Add the right definitions to the apex_flags target
  target_compile_definitions(apex_flags INTERFACE APEX_WITH_HIP)
  target_compile_definitions(apex_flags INTERFACE __HIP_PLATFORM_HCC__)

  list(APPEND _apex_imported_targets roctracer)
  list(APPEND _apex_imported_targets rocprofiler)
  list(APPEND _apex_imported_targets roctx)
  list(APPEND _apex_imported_targets rsmi)
else()
  add_custom_target(project_hip)
endif()

