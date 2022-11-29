# - Try to find LibROCTX
# Once done this will define
#  ROCTX_FOUND - System has ROCTX
#  ROCTX_INCLUDE_DIRS - The ROCTX include directories
#  ROCTX_LIBRARIES - The libraries needed to use ROCTX
#  ROCTX_DEFINITIONS - Compiler switches required for using ROCTX

if(NOT DEFINED $ROCTX_ROOT)
	if(DEFINED ENV{ROCTX_ROOT})
		# message("   env ROCTX_ROOT is defined as $ENV{ROCTX_ROOT}")
		set(ROCTX_ROOT $ENV{ROCTX_ROOT})
	endif()
	if(DEFINED $ROCTX_PATH)
		set(ROCTX_ROOT ${ROCTX_PATH})
	endif()
endif()

find_path(ROCTX_INCLUDE_DIR NAMES roctx.h
	HINTS ${ROCM_ROOT}/include/roctracer ${ROCTX_ROOT}/include)

find_library(ROCTX_LIBRARY NAMES roctx64
    HINTS ${ROCM_ROOT}/roctracer/lib64 ${ROCM_ROOT}/roctracer/lib ${ROCTX_ROOT}/lib64 ${ROCTX_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCTX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCTX  DEFAULT_MSG
                                  ROCTX_LIBRARY ROCTX_INCLUDE_DIR)

mark_as_advanced(ROCTX_INCLUDE_DIR ROCTX_LIBRARY)

if(ROCTX_FOUND)
  set(ROCTX_LIBRARIES ${CUDA_LIBRARY} ${ROCTX_LIBRARY} )
  set(ROCTX_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} ${ROCTX_INCLUDE_DIR})
  set(ROCTX_DIR ${ROCTX_ROOT})
  add_definitions(-DAPEX_HAVE_ROCTX)
endif()

