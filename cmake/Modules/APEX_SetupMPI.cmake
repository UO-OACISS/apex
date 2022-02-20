#  Copyright (c) 2014 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Setup an imported target for mpi
if(APEX_WITH_MPI)

  find_package(MPI)
  if(NOT MPI_FOUND)
      hpx_error("apex" "Requested APEX_WITH_MPI but could not find MPI.")
  endif()
  hpx_info("apex" "Building APEX with MPI support.")

  # Add an imported target
  add_library(mpi INTERFACE IMPORTED)
  set_property(TARGET mpi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${MPI_INCLUDE_DIR})
  set_property(TARGET mpi PROPERTY
    INTERFACE_LINK_LIBRARIES ${MPI_LIBRARIES})

  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${MPI_LIBRARY_DIR})
  target_compile_definitions(apex_flags INTERFACE APEX_HAVE_MPI)

  list(APPEND _apex_imported_targets mpi)
  add_definitions(-DAPEX_HAVE_MPI)

endif()
