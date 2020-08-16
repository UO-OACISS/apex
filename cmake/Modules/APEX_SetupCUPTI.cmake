#  Copyright (c) 2020 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED QUIET COMPONENTS CUPTI NVML)
  find_package(CUPTI REQUIRED QUIET COMPONENTS CUPTI)
  find_package(NVML REQUIRED QUIET COMPONENTS NVML)

  # Add an imported target
  add_library(cuda INTERFACE IMPORTED)
  add_library(cupti INTERFACE IMPORTED)
  add_library(nvidia-ml INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with CUDA/CUPTI support.")
  set_property(TARGET cupti PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${CUPTI_INCLUDE_DIRS})
  set_property(TARGET cuda PROPERTY
    INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARY})
  set_property(TARGET cupti PROPERTY
    INTERFACE_LINK_LIBRARIES ${CUPTI_LIBRARIES})

  set_property(TARGET nvidia-ml PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${NVML_INCLUDE_DIRS})
  set_property(TARGET nvidia-ml PROPERTY
    INTERFACE_LINK_LIBRARIES ${NVML_LIBRARIES})

  # Add the right definitions to the apex_flags target
  target_compile_definitions(apex_flags INTERFACE APEX_WITH_CUDA)

  list(APPEND _apex_imported_targets cudart)
  list(APPEND _apex_imported_targets cupti)
  list(APPEND _apex_imported_targets nvidia-ml)

else()
  add_custom_target(project_cuda)
endif()

