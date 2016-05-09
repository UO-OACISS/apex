# - Try to find LibSOS
# Once done this will define
#  SOS_FOUND - System has SOS
#  SOS_INCLUDE_DIRS - The SOS include directories
#  SOS_LIBRARIES - The libraries needed to use SOS
#  SOS_DEFINITIONS - Compiler switches required for using SOS

if(NOT DEFINED $SOS_ROOT)
	if(DEFINED ENV{SOS_ROOT})
		message("   env SOS_ROOT is defined as $ENV{SOS_ROOT}")
		set(SOS_ROOT $ENV{SOS_ROOT})
	endif()
endif()

find_path(SOS_INCLUDE_DIR NAMES sos.h
	HINTS ${SOS_ROOT}/include $ENV{SOS_ROOT}/include ${SOS_ROOT}/src $ENV{SOS_ROOT}/src)

find_library(SOS_LIBRARY NAMES sos HINTS ${SOS_ROOT}/lib $ENV{SOS_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set SOS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(SOS  DEFAULT_MSG
                                  SOS_LIBRARY SOS_INCLUDE_DIR)

mark_as_advanced(SOS_INCLUDE_DIR SOS_LIBRARY)

if(SOS_FOUND)
  set(SOS_LIBRARIES ${SOS_LIBRARY} )
  set(SOS_INCLUDE_DIRS ${SOS_INCLUDE_DIR})
  set(SOS_DIR ${SOS_ROOT})
  add_definitions(-DAPEX_HAVE_SOS)
endif()

