# - Try to find LibACTIVEHARMONY
# Once done this will define
#  ACTIVEHARMONY_FOUND - System has ACTIVEHARMONY
#  ACTIVEHARMONY_INCLUDE_DIRS - The ACTIVEHARMONY include directories
#  ACTIVEHARMONY_LIBRARIES - The libraries needed to use ACTIVEHARMONY
#  ACTIVEHARMONY_DEFINITIONS - Compiler switches required for using ACTIVEHARMONY

if(NOT DEFINED $ACTIVEHARMONY_ROOT)
    if(DEFINED ENV{ACTIVEHARMONY_ROOT})
        set(ACTIVEHARMONY_ROOT $ENV{ACTIVEHARMONY_ROOT})
    endif()
endif()

find_path(ACTIVEHARMONY_INCLUDE_DIR NAMES hclient.h
    HINTS ${ACTIVEHARMONY_ROOT}/include ${CMAKE_INSTALL_PREFIX}/ah/include)

find_library(ACTIVEHARMONY_LIBRARY NAMES harmony
    HINTS ${ACTIVEHARMONY_ROOT}/lib ${CMAKE_INSTALL_PREFIX}/ah/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ACTIVEHARMONY_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ACTIVEHARMONY  DEFAULT_MSG
                                  ACTIVEHARMONY_LIBRARY ACTIVEHARMONY_INCLUDE_DIR)

mark_as_advanced(ACTIVEHARMONY_INCLUDE_DIR ACTIVEHARMONY_LIBRARY)

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if((APEX_BUILD_ACTIVEHARMONY OR (NOT ACTIVEHARMONY_FOUND)) AND NOT APPLE)
  set(ACTIVEHARMONY_ROOT ${CMAKE_INSTALL_PREFIX}/ah CACHE STRING "Active Harmony Root directory")
  message("Attention: Downloading and Building ActiveHarmony as external project!")
  message(INFO " A working internet connection is required!")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  include(ExternalProject)
  ExternalProject_Add(project_activeharmony
    URL http://www.dyninst.org/sites/default/files/downloads/harmony/ah-4.6.0.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.6.0
    PATCH_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.6.0/src/project_activeharmony && patch -p0 code-server/code_generator.cxx < ${CMAKE_SOURCE_DIR}/cmake/Modules/ActiveHarmony.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.6.0/src/project_activeharmony && make -j${MAKEJOBS} MPICC=mpicc_disabled CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXXFLAGS=${CMAKE_CXX_FLAGS} LDFLAGS=${CMAKE_C_FLAGS}
    INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.6.0/src/project_activeharmony && make MPICC=mpicc_disabled CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXXFLAGS=${CMAKE_CXX_FLAGS} LDFLAGS=${CMAKE_C_FLAGS} install prefix=${ACTIVEHARMONY_ROOT}
    INSTALL_DIR ${ACTIVEHARMONY_ROOT}
    # LOG_DOWNLOAD 1
    # LOG_CONFIGURE 1
    # LOG_BUILD 1
    # LOG_INSTALL 1
  )
  #ExternalProject_Get_Property(project_activeharmony install_dir)
  add_library(harmony STATIC IMPORTED)
  set_property(TARGET harmony PROPERTY IMPORTED_LOCATION ${ACTIVEHARMONY_ROOT}/lib/libharmony.a)
  set(ACTIVEHARMONY_INCLUDE_DIR "${ACTIVEHARMONY_ROOT}/include")
  set(ACTIVEHARMONY_LIBRARY "${ACTIVEHARMONY_ROOT}/lib/libharmony.a")
  # handle the QUIETLY and REQUIRED arguments and set ACTIVEHARMONY_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(ACTIVEHARMONY  DEFAULT_MSG
                                    ACTIVEHARMONY_LIBRARY ACTIVEHARMONY_INCLUDE_DIR)
  set(ACTIVEHARMONY_FOUND TRUE)
else()
  add_custom_target(project_activeharmony)
endif()
# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(ACTIVEHARMONY_FOUND)
  set(ACTIVEHARMONY_LIBRARIES ${ACTIVEHARMONY_LIBRARY} )
  set(ACTIVEHARMONY_INCLUDE_DIRS ${ACTIVEHARMONY_INCLUDE_DIR})
  set(ACTIVEHARMONY_DIR ${ACTIVEHARMONY_ROOT})
  add_definitions(-DAPEX_HAVE_ACTIVEHARMONY)
endif()

