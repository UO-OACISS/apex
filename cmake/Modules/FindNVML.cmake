# - Try to find LibNVML
# Once done this will define
#  NVML_FOUND - System has NVML
#  NVML_INCLUDE_DIRS - The NVML include directories
#  NVML_LIBRARIES - The libraries needed to use NVML
#  NVML_DEFINITIONS - Compiler switches required for using NVML

if(NOT DEFINED $NVML_ROOT)
	if(DEFINED ENV{NVML_ROOT})
		 message("   env NVML_ROOT is defined as $ENV{NVML_ROOT}")
		set(NVML_ROOT $ENV{NVML_ROOT})
	endif()
endif()

set(NVML_ARCH_ROOT ${NVML_ROOT}/targets/x86_64-linux)

#if(NOT DEFINED $NVML_ROOT AND CUDAToolkit_FOUND)
#    message(INFO "   env NVML_ROOT is assuming ${CUDAToolkit_INCLUDE_DIRS}/..")
#    set(NVML_ROOT "${CUDAToolkit_INCLUDE_DIRS}/../extras/NVML")
#endif()

find_path(NVML_INCLUDE_DIR NAMES nvml.h
	HINTS ${CUDAToolkit_INCLUDE_DIRS} ${NVML_ROOT}/include ${NVML_ARCH_ROOT}/include)

find_library(NVML_LIBRARY NAMES nvml nvidia-ml
	HINTS ${CUDAToolkit_LIBRARY_DIR} ${NVML_ROOT} ${NVML_ROOT}/lib64 ${NVML_ROOT}/lib ${NVML_ARCH_ROOT} ${NVML_ARCH_ROOT}/lib64 ${NVML_ARCH_ROOT}/lib ${NVML_ROOT}/lib64/stubs ${NVML_ROOT}/lib/stubs)

find_library(CUDA_LIBRARY NAMES cudart
    HINTS ${CUDAToolkit_LIBRARY_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NVML_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(NVML  DEFAULT_MSG
                                  NVML_LIBRARY NVML_INCLUDE_DIR)

mark_as_advanced(NVML_INCLUDE_DIR NVML_LIBRARY)

if(NVML_FOUND)
  set(NVML_LIBRARIES ${CUDA_LIBRARY} ${NVML_LIBRARY} )
  set(NVML_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} ${NVML_INCLUDE_DIR})
  set(NVML_DIR ${NVML_ROOT})
  add_definitions(-DAPEX_HAVE_NVML)
endif()

