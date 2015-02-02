# - Try to find LibBFD
# Once done this will define
#  BFD_FOUND - System has BFD
#  BFD_INCLUDE_DIRS - The BFD include directories
#  BFD_LIBRARIES - The libraries needed to use BFD
#  BFD_DEFINITIONS - Compiler switches required for using BFD

find_package(PkgConfig)

# This if statement is specific to BFD, and should not be copied into other
# Find cmake scripts.

if(NOT BFD_ROOT AND NOT $ENV{BFD_ROOT} STREQUAL "")
	set(BFD_ROOT $ENV{BFD_ROOT})
endif()

pkg_check_modules(PC_BFD QUIET BFD)
set(BFD_DEFINITIONS ${PC_BFD_CFLAGS_OTHER})

find_path(BFD_INCLUDE_DIR bfd.h
          HINTS ${PC_BFD_INCLUDEDIR} ${PC_BFD_INCLUDE_DIRS} ${BFD_ROOT}/include
          PATH_SUFFIXES BFD )

set(TMP_PATH $ENV{LD_LIBRARY_PATH})
if ($TM_PATH)
	  string(REPLACE ":" " " LD_LIBRARY_PATH_STR $TMP_PATH)
endif()
find_library(BFD_LIBRARY NAMES bfd z
             HINTS ${PC_BFD_LIBDIR} ${PC_BFD_LIBRARY_DIRS} ${BFD_ROOT}/lib ${LD_LIBRARY_PATH_STR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BFD_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(BFD  DEFAULT_MSG
                                  BFD_LIBRARY BFD_INCLUDE_DIR)

mark_as_advanced(BFD_INCLUDE_DIR BFD_LIBRARY)

if(BFD_FOUND)
  set(BFD_LIBRARIES ${BFD_LIBRARY} )
  set(BFD_INCLUDE_DIRS ${BFD_INCLUDE_DIR})
  set(BFD_DIR ${BFD_ROOT})
  add_definitions(-DAPEX_HAVE_BFD)
endif()

