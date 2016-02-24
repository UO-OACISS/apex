# - Try to find LibDEMANGLE
# Once done this will define
#  DEMANGLE_FOUND - System has DEMANGLE
#  DEMANGLE_INCLUDE_DIRS - The DEMANGLE include directories
#  DEMANGLE_LIBRARIES - The libraries needed to use DEMANGLE
#  DEMANGLE_DEFINITIONS - Compiler switches required for using DEMANGLE

find_package(PkgConfig)

# This if statement is specific to DEMANGLE, and should not be copied into other
# Find cmake scripts.

if(NOT DEMANGLE_ROOT AND NOT $ENV{DEMANGLE_ROOT} STREQUAL "")
	set(DEMANGLE_ROOT $ENV{DEMANGLE_ROOT})
endif()

if(NOT BFD_ROOT AND NOT $ENV{BFD_ROOT} STREQUAL "")
    set(BFD_ROOT $ENV{BFD_ROOT})
endif()

pkg_check_modules(PC_DEMANGLE QUIET DEMANGLE)
set(DEMANGLE_DEFINITIONS ${PC_DEMANGLE_CFLAGS_OTHER})

find_path(DEMANGLE_INCLUDE_DIR demangle.h
          HINTS ${PC_DEMANGLE_INCLUDEDIR} ${PC_DEMANGLE_INCLUDE_DIRS} 
          ${DEMANGLE_ROOT}/include ${BFD_ROOT}/include /usr/include 
          ${PC_DEMANGLE_INCLUDEDIR}/* ${PC_DEMANGLE_INCLUDE_DIRS}/* 
          ${DEMANGLE_ROOT}/* ${BFD_ROOT}/* /usr/include/*
          PATH_SUFFIXES DEMANGLE )

find_library(DEMANGLE_LIBRARY NAMES iberty
    HINTS ${PC_DEMANGLE_LIBDIR} ${PC_DEMANGLE_LIBRARY_DIRS} ${DEMANGLE_ROOT}/lib ${BFD_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DEMANGLE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(DEMANGLE  DEFAULT_MSG
                                  DEMANGLE_LIBRARY DEMANGLE_INCLUDE_DIR)

mark_as_advanced(DEMANGLE_INCLUDE_DIR DEMANGLE_LIBRARY)

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if(NOT DEMANGLE_FOUND AND NOT APPLE AND BUILDING_BFD)
  include(ExternalProject)
  ExternalProject_Get_Property(project_binutils install_dir)
  add_library(iberty STATIC IMPORTED)
  set_property(TARGET iberty PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libiberty.a)
  set(DEMANGLE_LIBRARIES "${BFD_ROOT}/lib/libiberty.a")
  set(DEMANGLE_INCLUDE_DIR "${BFD_ROOT}/include")
  set(DEMANGLE_DIR "${BFD_ROOT}")
  # handle the QUIETLY and REQUIRED arguments and set DEMANGLE_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(DEMANGLE  DEFAULT_MSG
      DEMANGLE_LIBRARY DEMANGLE_INCLUDE_DIR)
  set(DEMANGLE_FOUND TRUE)
  set(BUILDING_BFD TRUE)
endif()
# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(DEMANGLE_FOUND)
  set(DEMANGLE_LIBRARIES ${DEMANGLE_LIBRARY} )
  set(DEMANGLE_INCLUDE_DIRS ${DEMANGLE_INCLUDE_DIR})
  set(DEMANGLE_DIR ${DEMANGLE_ROOT})
  add_definitions(-DHAVE_GNU_DEMANGLE)
else()
  unset(DEMANGLE_LIBRARY)
  unset(DEMANGLE_LIBRARIES)
  unset(DEMANGLE_DIR)
endif()

