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

find_path(ROCTRACER_INCLUDE_DIR NAMES roctracer.h
	HINTS ${ROCTRACER_ROOT}/include ${ROCM_ROOT}/include/roctracer)

find_path(HSA_INCLUDE_DIR NAMES hsa.h
	HINTS ${ROCM_ROOT}/include/hsa)

find_path(HIP_INCLUDE_DIR NAMES hip/hip_runtime_api.h
	HINTS ${ROCM_ROOT}/hip/include)

find_library(ROCTRACER_LIBRARY NAMES roctracer64
    HINTS ${ROCTRACER_ROOT}/lib64 ${ROCTRACER_ROOT}/lib ${ROCM_ROOT}/roctracer/lib64 ${ROCM_ROOT}/roctracer/lib)

#find_library(ROCTRACER_LIBRARY_KFD NAMES kfdwrapper64
    #HINTS ${ROCTRACER_ROOT}/lib64 ${ROCTRACER_ROOT}/lib ${ROCM_ROOT}/roctracer/lib64 ${ROCM_ROOT}/roctracer/lib)

find_library(HIP_LIBRARY NAMES amdhip64
    HINTS ${ROCM_ROOT}/hip/lib64 ${ROCM_ROOT}/hip/lib ${ROCM_ROOT}/lib64 ${ROCM_ROOT}/lib)

find_library(HSA_LIBRARY NAMES hsa-runtime64
    HINTS ${ROCM_ROOT}/hsa/lib64 ${ROCM_ROOT}/hsa/lib ${ROCM_ROOT}/lib64 ${ROCM_ROOT}/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCTRACER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCTRACER  DEFAULT_MSG
                                  ROCTRACER_LIBRARY HIP_LIBRARY HSA_LIBRARY
                                  ROCTRACER_INCLUDE_DIR HSA_INCLUDE_DIR HIP_INCLUDE_DIR)

mark_as_advanced(ROCTRACER_INCLUDE_DIR HSA_INCLUDE_DIR HIP_INCLUDE_DIR ROCTRACER_LIBRARY HIP_LIBRARY)

if(ROCTRACER_FOUND)
  set(ROCTRACER_LIBRARIES ${ROCTRACER_LIBRARY} ${HIP_LIBRARY} ${HSA_LIBRARY})
  set(ROCTRACER_INCLUDE_DIRS ${ROCTRACER_INCLUDE_DIR} ${HSA_INCLUDE_DIR} ${HIP_INCLUDE_DIR})
  set(ROCTRACER_DIR ${ROCTRACER_ROOT})
  add_definitions(-DAPEX_HAVE_ROCTRACER)
endif()

