# - Try to find LibPAPI
# Once done this will define
#  PAPI_FOUND - System has PAPI
#  PAPI_INCLUDE_DIRS - The PAPI include directories
#  PAPI_LIBRARIES - The libraries needed to use PAPI
#  PAPI_DEFINITIONS - Compiler switches required for using PAPI

if(NOT DEFINED $PAPI_ROOT)
	if(DEFINED ENV{PAPI_ROOT})
		# message("   env PAPI_ROOT is defined as $ENV{PAPI_ROOT}")
		set(PAPI_ROOT $ENV{PAPI_ROOT})
	endif()
endif()

find_path(PAPI_INCLUDE_DIR NAMES papi.h
	HINTS ${PAPI_ROOT}/include $ENV{PAPI_ROOT}/include)

find_library(PAPI_LIBRARY NAMES papi
	HINTS ${PAPI_ROOT}/* $ENV{PAPI_ROOT}/*)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PAPI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PAPI  DEFAULT_MSG
                                  PAPI_LIBRARY PAPI_INCLUDE_DIR)

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY)

if(PAPI_FOUND)
  set(PAPI_LIBRARIES ${PAPI_LIBRARY} )
  set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR})
  set(PAPI_DIR ${PAPI_ROOT})
  add_definitions(-DAPEX_HAVE_PAPI)
endif()

