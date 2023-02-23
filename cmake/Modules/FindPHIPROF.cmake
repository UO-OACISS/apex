# - Try to find PhiProf
# Once done this will define
#  PHIPROF_FOUND - System has PhiProf
#  PHIPROF_INCLUDE_DIRS - The  include directories


find_package(PkgConfig)

if(NOT DEFINED $PHIPROF_ROOT)
	if(DEFINED ENV{PHIPROF_ROOT})
		# message("   env PHIPROF_ROOT is defined as $ENV{PHIPROF_ROOT}")
		set(PHIPROF_ROOT $ENV{PHIPROF_ROOT})
	endif()
endif()

if(NOT DEFINED $PHIPROF_ROOT)
	if(DEFINED PHIPROF_DIR)
		# message("   env PHIPROF_ROOT is defined as $ENV{PHIPROF_ROOT}")
		set(PHIPROF_ROOT $PHIPROF_DIR)
	endif()
endif()

message(INFO " will check ${PHIPROF_ROOT} for PHIPROF")
set(CMAKE_PREFIX_PATH} "${CMAKE_PREFIX_PATH} ${PHIPROF_ROOT}")
#set(ENV{PKG_CONFIG_PATH} "${PHIPROF_ROOT}/libs/pkgconfig")

if(NOT PHIPROF_FOUND)
    find_path(PHIPROF_INCLUDE_DIR NAMES phiprof.hpp
	    HINTS ${PHIPROF_ROOT}/include)

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set PHIPROF_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(PHIPROF  DEFAULT_MSG
                                      PHIPROF_INCLUDE_DIR)

    mark_as_advanced(PHIPROF_INCLUDE_DIR)
    message( "Using ${PHIPROF_INCLUDE_DIR} as PhiProf include dir" )

    if(PHIPROF_FOUND)
        set(PHIPROF_INCLUDE_DIRS ${PHIPROF_INCLUDE_DIR})
        set(PHIPROF_DIR ${PHIPROF_ROOT})
        add_definitions(-DAPEX_HAVE_PHIPROF)
    endif()
endif()

