# - Try to find LibCUPTI
# Once done this will define
#  CUPTI_FOUND - System has CUPTI
#  CUPTI_INCLUDE_DIRS - The CUPTI include directories
#  CUPTI_LIBRARIES - The libraries needed to use CUPTI
#  CUPTI_DEFINITIONS - Compiler switches required for using CUPTI

if(NOT DEFINED $CUPTI_ROOT)
	if(DEFINED ENV{CUPTI_ROOT})
		# message("   env CUPTI_ROOT is defined as $ENV{CUPTI_ROOT}")
		set(CUPTI_ROOT $ENV{CUPTI_ROOT})
	endif()
endif()

if(NOT DEFINED $CUPTI_ROOT AND CUDAToolkit_FOUND)
    message(INFO "   env CUPTI_ROOT is assuming ${CUDAToolkit_INCLUDE_DIRS}/..")
    set(CUPTI_ROOT "${CUDAToolkit_INCLUDE_DIRS}/../extras/CUPTI")
endif()

find_path(CUPTI_INCLUDE_DIR NAMES cupti.h
	HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUPTI_ROOT}/include)

find_library(CUPTI_LIBRARY NAMES cupti
    HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUPTI_ROOT} ${CUPTI_ROOT}/lib64 ${CUPTI_ROOT}/lib)

find_library(CUDA_LIBRARY NAMES cudart
    HINTS ${CUDAToolkit_LIBRARY_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set CUPTI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CUPTI  DEFAULT_MSG
                                  CUPTI_LIBRARY CUPTI_INCLUDE_DIR)

mark_as_advanced(CUPTI_INCLUDE_DIR CUPTI_LIBRARY)

if(CUPTI_FOUND)
  set(CUPTI_LIBRARIES ${CUDA_LIBRARY} ${CUPTI_LIBRARY} )
  set(CUPTI_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR})
  set(CUPTI_DIR ${CUPTI_ROOT})
  add_definitions(-DAPEX_HAVE_CUPTI)
endif()

