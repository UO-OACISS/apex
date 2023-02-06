# - Try to find LibLEVEL0
# Once done this will define
#  LEVEL0_FOUND - System has LEVEL0
#  LEVEL0_INCLUDE_DIRS - The LEVEL0 include directories
#  LEVEL0_LIBRARIES - The libraries needed to use LEVEL0
#  LEVEL0_DEFINITIONS - Compiler switches required for using LEVEL0

if(NOT DEFINED $LEVEL0_ROOT)
	if(DEFINED ENV{LEVEL0_ROOT})
		# message("   env LEVEL0_ROOT is defined as $ENV{LEVEL0_ROOT}")
		set(LEVEL0_ROOT $ENV{LEVEL0_ROOT})
	endif()
endif()

find_path(LEVEL0_INCLUDE_DIR NAMES level_zero/ze_api.h
	HINTS ${LEVEL0_ROOT}/include /usr ${LEVEL0_ROOT})

find_library(LEVEL0_LIBRARY NAMES ze_loader
    HINTS ${LEVEL0_ROOT} ${LEVEL0_ROOT}/lib64 ${LEVEL0_ROOT}/lib /usr/lib64 /usr/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LEVEL0_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LEVEL0  DEFAULT_MSG
                                  LEVEL0_LIBRARY LEVEL0_INCLUDE_DIR)

mark_as_advanced(LEVEL0_INCLUDE_DIR LEVEL0_LIBRARY)

if(LEVEL0_FOUND)
  set(LEVEL0_LIBRARIES ${LEVEL0_LIBRARY} )
  set(LEVEL0_INCLUDE_DIRS ${LEVEL0_INCLUDE_DIR})
  set(LEVEL0_DIR ${LEVEL0_ROOT})
  add_definitions(-DAPEX_HAVE_LEVEL0)
endif()

