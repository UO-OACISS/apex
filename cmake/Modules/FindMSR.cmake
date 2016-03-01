# - Try to find libmsr
# Once done this will define
#  MSR_FOUND - System has MSR
#  MSR_INCLUDE_DIRS - The MSR include directories
#  MSR_LIBRARIES - The libraries needed to use MSR
#  MSR_DEFINITIONS - Compiler switches required for using MSR

find_package(PkgConfig)

if(NOT MSR_ROOT AND NOT $ENV{MSR_ROOT} STREQUAL "")
  set(MSR_ROOT $ENV{MSR_ROOT})
endif()

message(INFO " will check ${MSR_ROOT} for MSR")

pkg_check_modules(PC_MSR QUIET MSR)
set(MSR_DEFINITIONS ${PC_MSR_CFLAGS_OTHER})

find_path(MSR_INCLUDE_DIR msr/msr_core.h
          HINTS ${PC_MSR_INCLUDEDIR} ${PC_MSR_INCLUDE_DIRS} ${MSR_ROOT}/include)

find_library(MSR_LIBRARY NAMES msr
             HINTS ${PC_MSR_LIBDIR} ${PC_MSR_LIBRARY_DIRS} ${MSR_ROOT}/lib 
			 ${MSR_ROOT}/lib/* NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MSR_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MSR  DEFAULT_MSG
                                  MSR_LIBRARY MSR_INCLUDE_DIR)

mark_as_advanced(MSR_INCLUDE_DIR MSR_LIBRARY)

if(MSR_FOUND)
  set(MSR_LIBRARIES ${MSR_LIBRARY} )
  set(MSR_INCLUDE_DIRS ${MSR_INCLUDE_DIR} )
  set(MSR_DIR ${MSR_ROOT})
  add_definitions(-DAPEX_HAVE_MSR)
endif()

