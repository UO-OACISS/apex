# - Try to find LibROCTRACER
# Once done this will define
#  ROCTRACER_FOUND - System has ROCTRACER
#  ROCTRACER_INCLUDE_DIRS - The ROCTRACER include directories
#  ROCTRACER_LIBRARIES - The libraries needed to use ROCTRACER
#  ROCTRACER_DEFINITIONS - Compiler switches required for using ROCTRACER

if(NOT DEFINED $ROCTRACER_ROOT)
	if(DEFINED ENV{ROCTRACER_ROOT})
		# message("   env ROCTRACER_ROOT is defined as $ENV{ROCTRACER_ROOT}")
		set(ROCTRACER_ROOT $ENV{ROCTRACER_ROOT})
	endif()
	if(DEFINED $ROCTRACER_PATH)
		set(ROCTRACER_ROOT ${ROCTRACER_PATH})
	endif()
endif()

if (NOT DEFINED $ROCM_ROOT)
    if(DEFINED $ROCM_PATH)
        set(ROCM_ROOT ${ROCM_PATH})
    endif()
endif()

find_path(ROCTRACER_KFD_INCLUDE_DIR NAMES roctracer_kfd.h
	HINTS ${ROCTRACER_ROOT}/include ${ROCM_ROOT}/include/roctracer)

find_library(ROCTRACER_KFD_LIBRARY NAMES kfdwrapper64
    HINTS ${ROCTRACER_ROOT}/lib64 ${ROCTRACER_ROOT}/lib ${ROCM_ROOT}/roctracer/lib64 ${ROCM_ROOT}/roctracer/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCTRACER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCTRACER_KFD DEFAULT_MSG
                                  ROCTRACER_KFD_LIBRARY
                                  ROCTRACER_KFD_INCLUDE_DIR)

mark_as_advanced(ROCTRACER_KFD_INCLUDE_DIR ROCTRACER_KFD_LIBRARY)

if(ROCTRACER_KFD_FOUND)
  set(ROCTRACER_KFD_LIBRARIES ${ROCTRACER_KFD_LIBRARY})
  set(ROCTRACER_KFD_INCLUDE_DIRS ${ROCTRACER_KFD_INCLUDE_DIR})
  add_definitions(-DAPEX_HAVE_ROCTRACER_KFD)
endif()

