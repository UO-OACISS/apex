# - Find Tcmalloc
# Find the native Tcmalloc includes and library
#
#  Tcmalloc_INCLUDE_DIR - where to find Tcmalloc.h, etc.
#  Tcmalloc_LIBRARIES   - List of libraries when using Tcmalloc.
#  Tcmalloc_FOUND       - True if Tcmalloc found.

if (Tcmalloc_INCLUDE_DIR)
    # Already in cache, be silent
    set(Tcmalloc_FIND_QUIETLY TRUE)
endif ()

if(DEFINED GPERFTOOLS_ROOT)
    message(STATUS "GPERFTOOLS_ROOT: ${GPERFTOOLS_ROOT}")
    find_path(Tcmalloc_INCLUDE_DIR gperftools/malloc_hook.h ${GPERFTOOLS_ROOT}/include ${CMAKE_INSTALL_PREFIX}/gperftools/include)
    find_library(Tcmalloc_LIBRARY NAME tcmalloc PATHS ${GPERFTOOLS_ROOT}/lib ${CMAKE_INSTALL_PREFIX}/gperftools/lib  NO_DEFAULT_PATH)
else(DEFINED GPERFTOOLS_ROOT)
    find_path(Tcmalloc_INCLUDE_DIR gperftools/malloc_hook.h ${CMAKE_INSTALL_PREFIX}/gperftools/include)
    find_library(Tcmalloc_LIBRARY NAME tcmalloc PATHS ${CMAKE_INSTALL_PREFIX}/gperftools/lib  NO_DEFAULT_PATH)
    if (Tcmalloc_FOUND)
        set(Tcmalloc_ROOT ${CMAKE_INSTALL_PREFIX}/gperftools CACHE STRING "Google PerfTools Root directory")
    endif (Tcmalloc_FOUND)
endif(DEFINED GPERFTOOLS_ROOT)

if (Tcmalloc_INCLUDE_DIR AND Tcmalloc_LIBRARY)
    set(Tcmalloc_FOUND TRUE)
    set(Tcmalloc_LIBRARIES ${Tcmalloc_LIBRARY})
else ()
    set(Tcmalloc_FOUND FALSE)
    set(Tcmalloc_LIBRARIES)
endif ()

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if((APEX_BUILD_Tcmalloc OR (NOT Tcmalloc_FOUND)) AND NOT APPLE)
  set(Tcmalloc_ROOT ${CMAKE_INSTALL_PREFIX}/gperftools CACHE STRING "Google PerfTools Root directory")
  message("Attention: Downloading and Building Google PerfTools as external project!")
  message(INFO " A working internet connection is required!")
  include(ExternalProject)
  ExternalProject_Add(project_gperftools
    URL https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gperftools-2.9.1
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/gperftools-2.9.1/src/project_gperftools && ./configure --enable-shared --prefix=${Tcmalloc_ROOT}
    BUILD_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/gperftools-2.9.1/src/project_gperftools && make -j${MAKEJOBS}
    INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/gperftools-2.9.1/src/project_gperftools && make install
    INSTALL_DIR ${Tcmalloc_ROOT}
  )
  add_library(tcmalloc STATIC IMPORTED)
  set_property(TARGET tcmalloc PROPERTY IMPORTED_LOCATION ${Tcmalloc_ROOT}/lib/libtcmalloc.so)
  set(Tcmalloc_INCLUDE_DIR "${Tcmalloc_ROOT}/include")
  set(Tcmalloc_LIBRARY "${Tcmalloc_ROOT}/lib/libtcmalloc.so")
  #set(GPROFILER_LIBRARY "${Tcmalloc_ROOT}/lib/libprofiler.so")
  set(Tcmalloc_FOUND TRUE)
else()
  add_custom_target(project_gperftools)
endif()
# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ACTIVEHARMONY_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Tcmalloc DEFAULT_MSG
                                  Tcmalloc_LIBRARY Tcmalloc_INCLUDE_DIR)

mark_as_advanced(Tcmalloc_LIBRARY Tcmalloc_INCLUDE_DIR)

if (Tcmalloc_FOUND)
    if (NOT Tcmalloc_FIND_QUIETLY)
        message(STATUS "Found Tcmalloc: ${Tcmalloc_LIBRARY}")
    endif ()
else ()
    message(STATUS "Not Found Tcmalloc: ${Tcmalloc_LIBRARY}")
    if (Tcmalloc_FIND_REQUIRED)
        message(STATUS "Looked for Tcmalloc libraries named tcmalloc.")
        message(FATAL_ERROR "Could NOT find Tcmalloc library")
    endif ()
endif ()

