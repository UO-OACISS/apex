# - Try to find LibTAU
# Once done this will define
#  TAU_FOUND - System has TAU
#  TAU_INCLUDE_DIRS - The TAU include directories
#  TAU_LIBRARIES - The libraries needed to use TAU
#  TAU_DEFINITIONS - Compiler switches required for using TAU

find_package(PkgConfig)

# This if statement is specific to TAU, and should not be copied into other
# Find cmake scripts.
if(NOT TAU_ROOT AND NOT $ENV{HOME_TAU} STREQUAL "")
  set(TAU_ROOT $ENV{HOME_TAU})
endif()

if(NOT TAU_ARCH AND NOT $ENV{TAU_ARCH} STREQUAL "")
    set(TAU_ARCH $ENV{TAU_ARCH})
endif()

pkg_check_modules(PC_TAU QUIET TAU)
set(TAU_DEFINITIONS ${PC_TAU_CFLAGS_OTHER})

find_path(TAU_INCLUDE_DIR TAU.h
          HINTS ${TAU_ROOT}/include
          PATH_SUFFIXES TAU )

if (${APPLE})
    find_library(TAU_LIBRARY NAMES TAUsh${TAU_OPTIONS} tau${TAU_OPTIONS} TAU 
             HINTS ${TAU_ROOT}/apple/lib)
         find_path(TAU_LIBRARY_DIR NAMES TAUsh${TAU_OPTIONS}.so tau${TAU_OPTIONS}.a libTAU.dylib
             HINTS ${TAU_ROOT}/apple/lib)
else()
    find_library(TAU_LIBRARY NAMES TAUsh${TAU_OPTIONS} tau${TAU_OPTIONS} TAU 
      HINTS ${TAU_ROOT}/${TAU_ARCH}/lib ${TAU_ROOT}/${CMAKE_SYSTEM_PROCESSOR}/lib  ${TAU_ROOT}/*/lib )
  find_path(TAU_LIBRARY_DIR NAMES tau${TAU_OPTIONS}.a TAUsh${TAU_OPTIONS}.so libTAU.so libTAU.a libTAU.dylib
      HINTS ${TAU_ROOT}/${TAU_ARCH}/lib ${TAU_ROOT}/${CMAKE_SYSTEM_PROCESSOR}/lib  ${TAU_ROOT}/*/lib )
endif()

find_path(TAU_LIBRARY_DIR_2 NAMES libTauPthreadWrap.a
             HINTS ${TAU_LIBRARY_DIR}/static-pthread)

if (TAU_LIBRARY_DIR_2_FOUND)
    set(TAU_LIBRARIES ${TAU_LIBRARY} -L${TAU_LIBRARY_DIR_2} -Wl,-wrap,pthread_create -Wl,-wrap,pthread_join -Wl,-wrap,pthread_exit -Wl,-wrap,pthread_barrier_wait -lTauPthreadWrap )
else()
    set(TAU_LIBRARIES ${TAU_LIBRARY})
endif()

set(TAU_INCLUDE_DIRS ${TAU_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TAU_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(TAU  DEFAULT_MSG
                                  TAU_LIBRARY TAU_INCLUDE_DIR TAU_LIBRARIES )

mark_as_advanced(TAU_INCLUDE_DIR TAU_LIBRARY TAU_LIBRARIES )
set(TAU_DIR ${TAU_ROOT})

if(TAU_FOUND)
  add_definitions(-DAPEX_HAVE_TAU)
  message("TAU includes: ${TAU_INCLUDE_DIR}")
  message("TAU libraries: ${TAU_LIBRARIES}")
endif()

