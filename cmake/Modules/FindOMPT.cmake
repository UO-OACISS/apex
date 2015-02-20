# - Try to find LibOMPT
# Once done this will define
#  OMPT_FOUND - System has OMPT
#  OMPT_INCLUDE_DIRS - The OMPT include directories
#  OMPT_LIBRARIES - The libraries needed to use OMPT
#  OMPT_DEFINITIONS - Compiler switches required for using OMPT

find_package(PkgConfig)

# This if statement is specific to OMPT, and should not be copied into other
# Find cmake scripts.

if(NOT OMPT_ROOT AND NOT $ENV{OMPT_ROOT} STREQUAL "")
  set(OMPT_ROOT $ENV{OMPT_ROOT})
endif()

pkg_check_modules(PC_OMPT QUIET OMPT)
set(OMPT_DEFINITIONS ${PC_OMPT_CFLAGS_OTHER})

find_path(OMPT_INCLUDE_DIR ompt.h
          HINTS ${PC_OMPT_INCLUDEDIR} ${PC_OMPT_INCLUDE_DIRS} ${OMPT_ROOT}/include)

find_library(OMPT_LIBRARY NAMES iomp5
             HINTS ${PC_OMPT_LIBDIR} ${PC_OMPT_LIBRARY_DIRS} ${OMPT_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OMPT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OMPT  DEFAULT_MSG
                                  OMPT_LIBRARY OMPT_INCLUDE_DIR)

mark_as_advanced(OMPT_INCLUDE_DIR OMPT_LIBRARY)

if(OMPT_FOUND)
  set(OMPT_LIBRARIES ${OMPT_LIBRARY} )
  set(OMPT_INCLUDE_DIRS ${OMPT_INCLUDE_DIR} )
  set(OMPT_DIR ${OMPT_ROOT})
  add_definitions(-DAPEX_HAVE_OMPT)
endif()

