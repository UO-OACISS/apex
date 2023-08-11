
################################################################################
# Kokkos configuration - needed for HPX integration support
################################################################################

if(APEX_WITH_KOKKOS)
    message(INFO " Checking for Kokkos installation in $KOKKOS_ROOT...")
    find_package(Kokkos)
    if (Kokkos_FOUND)
        message(INFO " Using Kokkos include: ${Kokkos_INCLUDE_DIRS}/impl")
        include_directories(${Kokkos_INCLUDE_DIRS}/impl)
    else()
        message(INFO " Kokkos not found, cloning submodule to get required headers.")
        include(AddGitSubmodule)
        add_git_submodule(${CMAKE_CURRENT_SOURCE_DIR}/../../kokkos FALSE)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../kokkos/core/src/impl)
    endif()
endif()
