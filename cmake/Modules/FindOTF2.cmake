# - Try to find LibOTF2
# Once done this will define
#  OTF2_FOUND - System has OTF2
#  OTF2_INCLUDE_DIRS - The OTF2 include directories
#  OTF2_LIBRARIES - The libraries needed to use OTF2
#  OTF2_DEFINITIONS - Compiler switches required for using OTF2

if(NOT DEFINED OTF2_ROOT)
	if(DEFINED ENV{OTF2_ROOT})
		# message("   env OTF2_ROOT is defined as $ENV{OTF2_ROOT}")
		set(OTF2_ROOT $ENV{OTF2_ROOT})
	endif()
endif()

find_path(OTF2_INCLUDE_DIR NAMES otf2
	HINTS ${OTF2_ROOT}/include ${CMAKE_INSTALL_PREFIX}/otf2/include)

if(APPLE)
    find_library(OTF2_LIBRARY NAMES libotf2.a otf2
	    HINTS ${OTF2_ROOT}/lib ${CMAKE_INSTALL_PREFIX}/otf2/lib)
else()
    find_library(OTF2_LIBRARY NAMES otf2
	    HINTS ${OTF2_ROOT}/lib ${CMAKE_INSTALL_PREFIX}/otf2/lib)
endif(APPLE)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OTF2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OTF2  DEFAULT_MSG
                                  OTF2_LIBRARY OTF2_INCLUDE_DIR)

mark_as_advanced(OTF2_INCLUDE_DIR OTF2_LIBRARY)

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if(APEX_BUILD_OTF2 AND (NOT OTF2_FOUND))
  set(OTF2_ROOT ${CMAKE_INSTALL_PREFIX}/otf2 CACHE STRING "OTF2 Root directory" FORCE)
  message("Attention: Downloading and Building OTF2 as external project!")
  message(INFO " A working internet connection is required!")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  include(ExternalProject)
  ExternalProject_Add(project_otf2
    URL http://perftools.pages.jsc.fz-juelich.de/cicd/otf2/tags/otf2-2.3/otf2-2.3.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.3
    PATCH_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.3/src/project_otf2 && patch -p0 src/otf2_archive_int.c < ${PROJECT_SOURCE_DIR}/cmake/Modules/otf2_collective_callbacks.patch
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.3/src/project_otf2 && ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXXFLAGS=${CMAKE_CXX_FLAGS} LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} --prefix=${OTF2_ROOT} --enable-shared
    BUILD_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.3/src/project_otf2 && make -j${MAKEJOBS}
    INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/otf2-2.3/src/project_otf2 && make install
    INSTALL_DIR ${OTF2_ROOT}
    LOG_DOWNLOAD 1
    # LOG_CONFIGURE 1
    # LOG_BUILD 1
    # LOG_INSTALL 1
  )
  #ExternalProject_Get_Property(project_otf2 install_dir)
  add_library(otf2 STATIC IMPORTED)
  set_property(TARGET otf2 PROPERTY IMPORTED_LOCATION ${OTF2_ROOT}/lib/libotf2.a)
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

