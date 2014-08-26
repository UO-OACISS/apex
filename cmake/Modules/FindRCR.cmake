# - Try to find LibRCR
# Once done this will define
#  RCR_FOUND - System has RCR
#  RCR_INCLUDE_DIRS - The RCR include directories
#  RCR_LIBRARIES - The libraries needed to use RCR
#  RCR_DEFINITIONS - Compiler switches required for using RCR

find_package(PkgConfig)

# This if statement is specific to RCR, and should not be copied into other
# Find cmake scripts.

if(NOT RCR_ROOT AND NOT $ENV{RCR_ROOT} STREQUAL "")
  set(RCR_ROOT $ENV{RCR_ROOT})
endif()

pkg_check_modules(PC_RCR QUIET RCR)
set(RCR_DEFINITIONS ${PC_RCR_CFLAGS_OTHER})

find_path(RCR_INCLUDE_DIR RCRMSR.h
          HINTS ${PC_RCR_INCLUDEDIR} ${PC_RCR_INCLUDE_DIRS} ${RCR_ROOT}
          PATH_SUFFIXES RCR )

find_library(RCR_LIBRARY NAMES energyStat
             HINTS ${PC_RCR_LIBDIR} ${PC_RCR_LIBRARY_DIRS} ${RCR_ROOT})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RCR_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RCR  DEFAULT_MSG
                                  RCR_LIBRARY RCR_INCLUDE_DIR)

mark_as_advanced(RCR_INCLUDE_DIR RCR_LIBRARY)

if(RCR_FOUND)
  set(RCR_LIBRARIES ${RCR_LIBRARY} )
  set(RCR_INCLUDE_DIRS ${RCR_INCLUDE_DIR} )
  set(RCR_DIR ${RCR_ROOT})
  add_definitions(-DAPEX_HAVE_RCR)
endif()

