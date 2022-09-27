# - Try to find StarPU
# Once done this will define
#  STARPU_FOUND - System has StarPU
#  STARPU_INCLUDE_DIRS - The StarPU include directories
#  STARPU_LIBRARIES - The libraries needed to use StarPU
#  STARPU_DEFINITIONS - Compiler switches required for using StarPU

if(NOT DEFINED $STARPU_ROOT)
	if(DEFINED ENV{STARPU_ROOT})
		# message("   env STARPU_ROOT is defined as $ENV{STARPU_ROOT}")
		set(STARPU_ROOT $ENV{STARPU_ROOT})
	endif()
endif()

# TODO problem because lib name and include path contain the version number, ie libstarpu-1.3.a
      
find_path(STARPU_INCLUDE_DIR NAMES starpu.h
	HINTS ${STARPU_ROOT}/include/starpu/1.3/ $ENV{STARPU_ROOT}/include/starpu/1.3/)
      
if(APPLE)
    find_library(STARPU_LIBRARY NAMES libstarpu-1.3.a starpu-1.3
	    HINTS ${STARPU_ROOT}/* $ENV{STARPU_ROOT}/*)
else()
    find_library(STARPU_LIBRARY NAMES starpu-1.3
	    HINTS ${STARPU_ROOT}/* $ENV{STARPU_ROOT}/*)
endif(APPLE)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set STARPU_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(STARPU  DEFAULT_MSG
                                  STARPU_LIBRARY STARPU_INCLUDE_DIR)
#                                  STARPU_INCLUDE_DIR)

#mark_as_advanced(STARPU_INCLUDE_DIR STARPU_LIBRARY)
mark_as_advanced(STARPU_INCLUDE_DIR)
message( "Using ${STARPU_INCLUDE_DIR} as StarPU include dir" )

if(STARPU_FOUND)
  set(STARPU_LIBRARIES ${STARPU_LIBRARY} )
  set(STARPU_INCLUDE_DIRS ${STARPU_INCLUDE_DIR})
  set(STARPU_DIR ${STARPU_ROOT})
  add_definitions(-DAPEX_HAVE_STARPU)
endif()

