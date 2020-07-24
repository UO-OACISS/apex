#  Copyright (c) 2020 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED QUIET COMPONENTS CUPTI)

  # Add an imported target
  add_library(cuda INTERFACE IMPORTED)
  add_library(cupti INTERFACE IMPORTED)
  hpx_info("apex" "Building APEX with CUDA/CUPTI support.")
  set_property(TARGET cupti PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${CUDAToolkit_INCLUDE_DIR})
  set_property(TARGET cuda PROPERTY
    INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARY})
  set_property(TARGET cupti PROPERTY
    INTERFACE_LINK_LIBRARIES ${CUDA_cupti_LIBRARY})

  # Add the right definitions to the apex_flags target
  target_compile_definitions(apex_flags INTERFACE APEX_WITH_CUDA)

  list(APPEND _apex_imported_targets cuda)
  list(APPEND _apex_imported_targets cupti)

else()
  add_custom_target(project_cuda)
endif()

