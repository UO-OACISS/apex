# - Try to find libocr
# Once done this will define
#  OCR_FOUND - System has OCR
#  OCR_INCLUDE_DIRS - The OCR include directories
#  OCR_LIBRARIES - The libraries needed to use OCR
#  OCR_DEFINITIONS - Compiler switches required for using OCR

find_package(PkgConfig)

if(NOT OCR_ROOT AND NOT $ENV{OCR_ROOT} STREQUAL "")
  set(OCR_ROOT $ENV{OCR_ROOT})
endif()

message(INFO " will check ${OCR_ROOT} for OCR")

pkg_check_modules(PC_OCR QUIET OCR)
set(OCR_DEFINITIONS ${PC_OCR_CFLAGS_OTHER})

find_path(OCR_INCLUDE_DIR ocr.h
          HINTS ${PC_OCR_INCLUDEDIR} ${PC_OCR_INCLUDE_DIRS} ${OCR_ROOT}/include)

find_path(OCR_INTERNAL_INCLUDE_DIR ocr-runtime-types.h
          HINTS ${PC_OCR_INCLUDEDIR}/../../src/inc ${PC_OCR_INCLUDE_DIRS}/../../src/inc ${OCR_ROOT}/../src/inc)

find_path(OCR_SRC_DIR README
          HINTS ${PC_OCR_INCLUDEDIR}/../../src ${PC_OCR_INCLUDE_DIRS}/../../src ${OCR_ROOT}/../src)

find_path(OCR_BUILD_DIR ocr-config.h
          HINTS ${PC_OCR_INCLUDEDIR}/../../build/x86-mpi ${PC_OCR_INCLUDE_DIRS}/../../build/x86-mpi ${OCR_ROOT}/../build/x86-mpi)

find_library(OCR_LIBRARY NAMES ocr_x86
             HINTS ${PC_OCR_LIBDIR} ${PC_OCR_LIBRARY_DIRS} ${OCR_ROOT}/lib 
			 ${OCR_ROOT}/lib/* NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OCR_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OCR  DEFAULT_MSG
    OCR_LIBRARY OCR_INCLUDE_DIR OCR_INTERNAL_INCLUDE_DIR OCR_SRC_DIR OCR_BUILD_DIR)

mark_as_advanced(OCR_INCLUDE_DIR OCR_LIBRARY)

if(OCR_FOUND)
  set(OCR_LIBRARIES ${OCR_LIBRARY} )
  set(OCR_INCLUDE_DIRS ${OCR_INCLUDE_DIR} ${OCR_INTERNAL_INCLUDE_DIR} ${OCR_SRC_DIR} ${OCR_BUILD_DIR})
  set(OCR_DIR ${OCR_ROOT})
  set(APEX_HAVE_OCR TRUE)
  add_definitions(-DAPEX_HAVE_OCR)
endif()

