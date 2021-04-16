# - Try to find LibKokkos
# Once done this will define
#  Kokkos_FOUND - System has Kokkos
#  Kokkos_INCLUDE_DIRS - The Kokkos include directories
#  Kokkos_LIBRARIES - The libraries needed to use Kokkos
#  Kokkos_DEFINITIONS - Compiler switches required for using Kokkos

if(NOT DEFINED $Kokkos_ROOT)
	if(DEFINED ENV{Kokkos_ROOT})
		# message("   env Kokkos_ROOT is defined as $ENV{Kokkos_ROOT}")
		set(Kokkos_ROOT $ENV{Kokkos_ROOT})
	endif()
endif()

message("Kokkos_ROOT is defined as ${Kokkos_ROOT}")

find_path(Kokkos_INCLUDE_DIR NAMES Kokkos_Core.hpp
	HINTS ${Kokkos_ROOT}/include $ENV{Kokkos_ROOT}/include)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Kokkos_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Kokkos  DEFAULT_MSG
                                  Kokkos_INCLUDE_DIR)

mark_as_advanced(Kokkos_INCLUDE_DIR Kokkos_LIBRARY)

if(Kokkos_FOUND)
  set(Kokkos_INCLUDE_DIRS ${Kokkos_INCLUDE_DIR})
  set(Kokkos_DIR ${Kokkos_ROOT})
  add_definitions(-DAPEX_HAVE_KOKKOS)
endif()

