# - Try to find LibOTF2
# Once done this will define
#  OTF2_FOUND - System has OTF2
#  OTF2_INCLUDE_DIRS - The OTF2 include directories
#  OTF2_LIBRARIES - The libraries needed to use OTF2
#  OTF2_DEFINITIONS - Compiler switches required for using OTF2

if(NOT DEFINED $OTF2_ROOT)
	if(DEFINED ENV{OTF2_ROOT})
		# message("   env OTF2_ROOT is defined as $ENV{OTF2_ROOT}")
		set(OTF2_ROOT $ENV{OTF2_ROOT})
	endif()
endif()

find_path(OTF2_INCLUDE_DIR NAMES otf2
	HINTS ${OTF2_ROOT}/include $ENV{OTF2_ROOT}/include)

if(APPLE)
    find_library(OTF2_LIBRARY NAMES libotf2.a otf2
	    HINTS ${OTF2_ROOT}/* $ENV{OTF2_ROOT}/*)
else()
    find_library(OTF2_LIBRARY NAMES otf2
	    HINTS ${OTF2_ROOT}/* $ENV{OTF2_ROOT}/*)
endif(APPLE)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OTF2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OTF2  DEFAULT_MSG
                                  OTF2_LIBRARY OTF2_INCLUDE_DIR)

mark_as_advanced(OTF2_INCLUDE_DIR OTF2_LIBRARY)

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if(BUILD_OTF2 OR (NOT OTF2_FOUND))
  set(CACHE OTF2_ROOT ${CMAKE_INSTALL_PREFIX} STRING "OTF2 Root directory")
  message("Attention: Downloading and Building OTF2 as external project!")
  message(INFO " A working internet connection is required!")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  include(ExternalProject)
  ExternalProject_Add(project_otf2
    URL http://www.vi-hps.org/upload/packages/otf2/otf2-2.0.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.0
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.0/src/project_otf2 && ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXXFLAGS=${CMAKE_CXX_FLAGS} LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} --prefix=${CMAKE_INSTALL_PREFIX} --enable-shared
    BUILD_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.0/src/project_otf2 && make
    INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.0/src/project_otf2 && make install
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  set(OTF2_ROOT ${CMAKE_INSTALL_PREFIX})
  #ExternalProject_Get_Property(project_otf2 install_dir)
  add_library(otf2 STATIC IMPORTED)
  set_property(TARGET otf2 PROPERTY IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libotf2.a)
  set(OTF2_INCLUDE_DIR "${OTF2_ROOT}/include")
  set(OTF2_LIBRARY "${OTF2_ROOT}/lib/libotf2.a")
  # handle the QUIETLY and REQUIRED arguments and set OTF2_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(OTF2  DEFAULT_MSG
                                    OTF2_LIBRARY OTF2_INCLUDE_DIR)
  set(OTF2_FOUND TRUE)
else()
  add_custom_target(project_otf2)
endif()
# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(OTF2_FOUND)
  set(OTF2_LIBRARIES ${OTF2_LIBRARY} )
  set(OTF2_INCLUDE_DIRS ${OTF2_INCLUDE_DIR})
  set(OTF2_DIR ${OTF2_ROOT})
  add_definitions(-DAPEX_HAVE_OTF2)
endif()

