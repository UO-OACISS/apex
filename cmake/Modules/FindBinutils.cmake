# - Try to find LibBINUTILS
# Once done this will define
#  BINUTILS_FOUND - System has BINUTILS
#  BINUTILS_INCLUDE_DIRS - The BINUTILS include directories
#  BINUTILS_LIBRARIES - The libraries needed to use BINUTILS
#  BINUTILS_DEFINITIONS - Compiler switches required for using BINUTILS

find_package(PkgConfig)

# This if statement is specific to BINUTILS, and should not be copied into other
# Find cmake scripts.

if(NOT BINUTILS_ROOT AND NOT $ENV{BINUTILS_ROOT} STREQUAL "")
	set(BINUTILS_ROOT $ENV{BINUTILS_ROOT})
endif()

pkg_check_modules(PC_BINUTILS QUIET BINUTILS)
set(BINUTILS_DEFINITIONS ${PC_BINUTILS_CFLAGS_OTHER})

find_path(BINUTILS_INCLUDE_DIR bfd.h
          HINTS ${PC_BINUTILS_INCLUDEDIR} ${PC_BINUTILS_INCLUDE_DIRS} ${BINUTILS_ROOT}/include
          PATH_SUFFIXES BINUTILS )

set(TMP_PATH $ENV{LD_LIBRARY_PATH})
if ($TMP_PATH)
	  string(REPLACE ":" " " LD_LIBRARY_PATH_STR $TMP_PATH)
endif()
find_library(BINUTILS_LIBRARY NAMES bfd z
             HINTS ${PC_BINUTILS_LIBDIR} ${PC_BINUTILS_LIBRARY_DIRS} ${BINUTILS_ROOT}/lib ${LD_LIBRARY_PATH_STR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BINUTILS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(BINUTILS  DEFAULT_MSG
                                  BINUTILS_LIBRARY BINUTILS_INCLUDE_DIR)

mark_as_advanced(BINUTILS_INCLUDE_DIR BINUTILS_LIBRARY)

if(BINUTILS_FOUND)
  set(BINUTILS_LIBRARIES ${BINUTILS_LIBRARY} )
  set(BINUTILS_INCLUDE_DIRS ${BINUTILS_INCLUDE_DIR})
  set(BINUTILS_DIR ${BINUTILS_ROOT})
  add_definitions(-DAPEX_HAVE_BINUTILS)
endif()

