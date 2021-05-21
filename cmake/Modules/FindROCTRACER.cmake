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

find_path(ROCTRACER_INCLUDE_DIR NAMES roctracer.h
	HINTS ${ROCM_ROOT}/include/roctracer ${ROCTRACER_ROOT}/include)

find_library(ROCTRACER_LIBRARY NAMES roctracer64
    HINTS ${ROCM_ROOT}/lib64 ${ROCM_ROOT}/lib ${ROCTRACER_ROOT}/lib64 ${ROCTRACER_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCTRACER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCTRACER  DEFAULT_MSG
                                  ROCTRACER_LIBRARY ROCTRACER_INCLUDE_DIR)

mark_as_advanced(ROCTRACER_INCLUDE_DIR ROCTRACER_LIBRARY)

if(ROCTRACER_FOUND)
  set(ROCTRACER_LIBRARIES ${CUDA_LIBRARY} ${ROCTRACER_LIBRARY} )
  set(ROCTRACER_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} ${ROCTRACER_INCLUDE_DIR})
  set(ROCTRACER_DIR ${ROCTRACER_ROOT})
  add_definitions(-DAPEX_HAVE_ROCTRACER)
endif()

