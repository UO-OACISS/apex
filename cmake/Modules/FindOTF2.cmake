# - Try to find LibOTF2
# Once done this will define
#  OTF2_FOUND - System has OTF2
#  OTF2_INCLUDE_DIRS - The OTF2 include directories
#  OTF2_LIBRARIES - The libraries needed to use OTF2
#  OTF2_DEFINITIONS - Compiler switches required for using OTF2

if(NOT DEFINED $OTF2_ROOT)
	if(DEFINED ENV{OTF2_ROOT})
		# message("   env OTF2_ROOT is defined as $ENV{OTF2_ROOT}")
		set(OTF2_ROOT $ENV{OTF2_ROOT})
	endif()
endif()

find_path(OTF2_INCLUDE_DIR NAMES otf2
	HINTS ${OTF2_ROOT}/include $ENV{OTF2_ROOT}/include)

if(APPLE)
    find_library(OTF2_LIBRARY NAMES libotf2.a otf2
	    HINTS ${OTF2_ROOT}/* $ENV{OTF2_ROOT}/*)
else()
    find_library(OTF2_LIBRARY NAMES otf2
	    HINTS ${OTF2_ROOT}/* $ENV{OTF2_ROOT}/*)
endif(APPLE)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OTF2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OTF2  DEFAULT_MSG
                                  OTF2_LIBRARY OTF2_INCLUDE_DIR)

mark_as_advanced(OTF2_INCLUDE_DIR OTF2_LIBRARY)

add_custom_target(project_otf2)

if(OTF2_FOUND)
  set(OTF2_LIBRARIES ${OTF2_LIBRARY} )
  set(OTF2_INCLUDE_DIRS ${OTF2_INCLUDE_DIR})
  set(OTF2_DIR ${OTF2_ROOT})
  add_definitions(-DAPEX_HAVE_OTF2)
endif()

