# - Try to find LibROCPROFILER
# Once done this will define
#  ROCPROFILER_FOUND - System has ROCPROFILER
#  ROCPROFILER_INCLUDE_DIRS - The ROCPROFILER include directories
#  ROCPROFILER_LIBRARIES - The libraries needed to use ROCPROFILER
#  ROCPROFILER_DEFINITIONS - Compiler switches required for using ROCPROFILER

if(NOT DEFINED $ROCPROFILER_ROOT)
	if(DEFINED ENV{ROCPROFILER_ROOT})
		# message("   env ROCPROFILER_ROOT is defined as $ENV{ROCPROFILER_ROOT}")
		set(ROCPROFILER_ROOT $ENV{ROCPROFILER_ROOT})
	endif()
	if(DEFINED $ROCPROFILER_PATH)
		set(ROCPROFILER_ROOT ${ROCPROFILER_PATH})
	endif()
endif()

if (NOT DEFINED $ROCM_ROOT)
    if(DEFINED $ROCM_PATH)
        set(ROCM_ROOT ${ROCM_PATH})
    endif()
endif()

find_path(ROCPROFILER_INCLUDE_DIR NAMES rocprofiler.h
	HINTS ${ROCM_ROOT}/include/rocprofiler ${ROCPROFILER_ROOT}/include)

find_library(ROCPROFILER_LIBRARY NAMES rocprofiler64
    HINTS ${ROCM_ROOT}/lib64 ${ROCM_ROOT}/lib ${ROCPROFILER_ROOT}/lib64 ${ROCPROFILER_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCPROFILER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCPROFILER  DEFAULT_MSG
                                  ROCPROFILER_LIBRARY
                                  ROCPROFILER_INCLUDE_DIR)

mark_as_advanced(ROCPROFILER_INCLUDE_DIR ROCPROFILER_LIBRARY)

if(ROCPROFILER_FOUND)
  set(ROCPROFILER_LIBRARIES ${ROCPROFILER_LIBRARY})
  set(ROCPROFILER_INCLUDE_DIRS ${ROCPROFILER_INCLUDE_DIR})
  set(ROCPROFILER_DIR ${ROCPROFILER_ROOT})
  add_definitions(-DAPEX_HAVE_ROCPROFILER)
endif()

