# - Try to find LibBFD
# Once done this will define
#  BFD_FOUND - System has BFD
#  BFD_INCLUDE_DIRS - The BFD include directories
#  BFD_LIBRARIES - The libraries needed to use BFD
#  BFD_DEFINITIONS - Compiler switches required for using BFD

find_package(PkgConfig)

# This if statement is specific to BFD, and should not be copied into other
# Find cmake scripts.

if(NOT BFD_ROOT AND NOT $ENV{BFD_ROOT} STREQUAL "")
	set(BFD_ROOT $ENV{BFD_ROOT})
endif()

pkg_check_modules(PC_BFD QUIET BFD)
set(BFD_DEFINITIONS ${PC_BFD_CFLAGS_OTHER})

find_path(BFD_INCLUDE_DIR bfd.h
          HINTS ${BFD_ROOT}/include 
          ${PC_BFD_INCLUDEDIR} 
          ${PC_BFD_INCLUDE_DIRS}
          PATH_SUFFIXES BFD )

set(TMP_PATH $ENV{LD_LIBRARY_PATH})
if ($TMP_PATH)
	  string(REPLACE ":" " " LD_LIBRARY_PATH_STR $TMP_PATH)
endif()
find_library(BFD_LIBRARY NAMES bfd
             HINTS ${BFD_ROOT}/lib ${BFD_ROOT}/lib64
             ${PC_BFD_LIBDIR} 
             ${PC_BFD_LIBRARY_DIRS} 
             ${LD_LIBRARY_PATH_STR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BFD_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(BFD  DEFAULT_MSG
                                  BFD_LIBRARY BFD_INCLUDE_DIR)

mark_as_advanced(BFD_INCLUDE_DIR BFD_LIBRARY)

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if((APEX_BUILD_BFD OR (NOT BFD_FOUND)) AND NOT APPLE)
  message("Attention: Downloading and Building binutils as external project!")
  message(INFO " A working internet connection is required!")
  include(ExternalProject)
  ExternalProject_Add(project_binutils
    URL "http://ftp.gnu.org/gnu/binutils/binutils-2.25.tar.bz2"
    URL_HASH SHA256=22defc65cfa3ef2a3395faaea75d6331c6e62ea5dfacfed3e2ec17b08c882923
    CONFIGURE_COMMAND <SOURCE_DIR>/configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXXFLAGS=${CMAKE_CXX_FLAGS} LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} --prefix=${CMAKE_INSTALL_PREFIX} --disable-dependency-tracking --enable-interwork --disable-multilib --enable-shared --enable-64-bit-bfd --target=${TARGET_ARCH} --enable-install-libiberty
    BUILD_COMMAND make MAKEINFO=true -j${MAKEJOBS}
    INSTALL_COMMAND make MAKEINFO=true install
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  ExternalProject_Add_Step(project_binutils basedirs
    DEPENDEES install
    COMMAND cp <SOURCE_DIR>/include/demangle.h ${CMAKE_INSTALL_PREFIX}/include/.
    COMMENT "Copying additional headers"
  )

  set(BFD_ROOT ${CMAKE_INSTALL_PREFIX})
  ExternalProject_Get_Property(project_binutils install_dir)
  add_library(bfd STATIC IMPORTED)
  set_property(TARGET bfd PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libbfd.so)
  set(BFD_INCLUDE_DIR ${BFD_ROOT}/include ${BFD_ROOT}/include/libiberty)
  set(BFD_LIBRARY "${BFD_ROOT}/lib/libbfd.so")
  # handle the QUIETLY and REQUIRED arguments and set BFD_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(BFD  DEFAULT_MSG
                                    BFD_LIBRARY BFD_INCLUDE_DIR)
  set(BFD_FOUND TRUE)
  set(BUILDING_BFD TRUE) # this tells the FindDemangle module that we are building it
else()
  add_custom_target(project_binutils)
endif()
# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #

if(BFD_FOUND)
  set(BFD_LIBRARIES ${BFD_LIBRARY} )
  set(BFD_INCLUDE_DIRS ${BFD_INCLUDE_DIR})
  set(BFD_DIR ${BFD_ROOT})
  add_definitions(-DAPEX_HAVE_BFD)
endif()

