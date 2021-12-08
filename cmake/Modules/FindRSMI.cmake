# - Try to find LibRSMI
# Once done this will define
#  RSMI_FOUND - System has RSMI
#  RSMI_INCLUDE_DIRS - The RSMI include directories
#  RSMI_LIBRARIES - The libraries needed to use RSMI
#  RSMI_DEFINITIONS - Compiler switches required for using RSMI

if(NOT DEFINED $RSMI_ROOT)
	if(DEFINED ENV{RSMI_ROOT})
		# message("   env RSMI_ROOT is defined as $ENV{RSMI_ROOT}")
		set(RSMI_ROOT $ENV{RSMI_ROOT})
	endif()
endif()

if(NOT DEFINED $RSMI_ROOT AND ROCM_PATH)
    message(INFO "   env RSMI_ROOT is assuming ${ROCM_PATH}/rocm_smi")
    set(RSMI_ROOT "${ROCM_PATH}/rocm_smi")
endif()

find_path(RSMI_INCLUDE_DIR NAMES rocm_smi/rocm_smi.h
	HINTS ${RSMI_ROOT}/include ${ROCM_ROOT}/rocm_smi/include)

find_library(RSMI_LIBRARY NAMES rocm_smi64
    HINTS ${RSMI_ROOT} ${RSMI_ROOT}/lib64 ${RSMI_ROOT}/lib ${ROCM_ROOT}/rocm_smi/lib64 ${ROCM_ROOT}/rocm_smi/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RSMI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RSMI  DEFAULT_MSG
                                  RSMI_LIBRARY RSMI_INCLUDE_DIR)

mark_as_advanced(RSMI_INCLUDE_DIR RSMI_LIBRARY)

if(RSMI_FOUND)
  set(RSMI_LIBRARIES ${RSMI_LIBRARY} )
  set(RSMI_INCLUDE_DIRS ${RSMI_INCLUDE_DIR})
  set(RSMI_DIR ${RSMI_ROOT})
  add_definitions(-DAPEX_HAVE_RSMI)
endif()

